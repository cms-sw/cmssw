#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

// default constructor; must not be called
template<class TauType, class TauDiscriminator>
TauDiscriminationProducerBase<TauType, TauDiscriminator>::TauDiscriminationProducerBase()
{
   throw cms::Exception("TauDiscriminationProducerBase") << " -- default ctor called; derived classes must call " << 
      "TauDiscriminationProducerBase(const ParameterSet&)";
}

//--- standard constructor from PSet
template<class TauType, class TauDiscriminator>
TauDiscriminationProducerBase<TauType, TauDiscriminator>::TauDiscriminationProducerBase(const ParameterSet& iConfig)
{   
   // tau collection to discriminate
   TauProducer_        = iConfig.getParameter<InputTag>(getProducerString<TauType>());

   // prediscriminant operator 
   // require the tau to pass the following prediscriminants
   const ParameterSet& prediscriminantConfig = iConfig.getParameter<ParameterSet>("Prediscriminants");

   // determine boolean operator used on the prediscriminants
   string pdBoolOperator = prediscriminantConfig.getParameter<string>("BooleanOperator");
   // convert string to lowercase
   transform(pdBoolOperator.begin(), pdBoolOperator.end(), pdBoolOperator.begin(), ::tolower);

   if( pdBoolOperator == "and" )
   {
      andPrediscriminants_ = 0x1; //use chars instead of bools so we can do a bitwise trick later
   } 
   else if ( pdBoolOperator == "or" )
   {
      andPrediscriminants_ = 0x0;
   } 
   else
   {
      throw cms::Exception("TauDiscriminationProducerBase") << "PrediscriminantBooleanOperator defined incorrectly, options are: AND,OR";
   }

   // get the list of prediscriminants
   vector<string> prediscriminantsNames = prediscriminantConfig.getParameterNamesForType<ParameterSet>();

   for( vector<string>::const_iterator iDisc  = prediscriminantsNames.begin();
                                       iDisc != prediscriminantsNames.end(); ++iDisc )
   {
      const ParameterSet& iPredisc = prediscriminantConfig.getParameter<ParameterSet>(*iDisc);
      const InputTag& label        = iPredisc.getParameter<InputTag>("Producer");
      double cut                   = iPredisc.getParameter<double>("cut");

      TauDiscInfo thisDiscriminator;
      thisDiscriminator.label = label;
      thisDiscriminator.cut   = cut;
      prediscriminants_.push_back(thisDiscriminator);
   }
  
   prediscriminantFailValue_ = 0.;

   // register product
   produces<TauDiscriminator>();
}

template<class TauType, class TauDiscriminator>
void TauDiscriminationProducerBase<TauType, TauDiscriminator>::produce(Event& event, const EventSetup& eventSetup)
{
   // setup function - does nothing in base, but can be overridden to retrieve PV or other stuff
   beginEvent(event, eventSetup);

   // retrieve the tau collection to discriminate
   Handle<TauCollection> taus;
   event.getByLabel(TauProducer_, taus);

   // output product
   auto_ptr<TauDiscriminator> output(new TauDiscriminator(TauRefProd(taus)));

   size_t nTaus = taus->size();

   // load prediscriminators
   size_t nPrediscriminants = prediscriminants_.size();
   for( size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc )
   {
      prediscriminants_[iDisc].fill(event);
   }

   // loop over taus
   for( size_t iTau = 0; iTau < nTaus; ++iTau )
   {
      // get reference to tau
      TauRef tauRef(taus, iTau);

      bool passesPrediscriminants = true;
      // check tau passes prediscriminants
      for( size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc )
      {
         // current discriminant result for this tau
         double discResult  = (*prediscriminants_[iDisc].handle)[tauRef];
         uint8_t thisPasses = ( discResult > prediscriminants_[iDisc].cut ) ? 1 : 0;


         // if we are using the AND option, as soon as one fails, 
         // the result is FAIL and we can quit looping.  
         // if we are using the OR option as soon as one passes,
         // the result is pass and we can quit looping

         // truth table
         //        |   result (thisPasses)
         //        |     F     |     T
         //-----------------------------------
         // AND(T) | res=fails |  continue
         //        |  break    |    
         //-----------------------------------
         // OR (F) |  continue | res=passes
         //        |           |  break 
         
         if( thisPasses ^ andPrediscriminants_ ) //XOR
         {
            passesPrediscriminants = ( andPrediscriminants_ ? 0 : 1 ); //NOR
            break;
         }
      }

      double result = prediscriminantFailValue_;
      if( passesPrediscriminants )
      {
         // this tau passes the prereqs, call our implemented discrimination function
         result = discriminate(tauRef);
      }

      // store the result of this tau into our new discriminator
      output->setValue(iTau, result);
   }
   event.put(output);
}

// template specialiazation to get the correct (Calo/PF)TauProducer names
template<> string getProducerString<PFTau>()   { return "PFTauProducer"; }
template<> string getProducerString<CaloTau>() { return "CaloTauProducer"; }

// compile our desired types and make available to linker
template class TauDiscriminationProducerBase<PFTau, PFTauDiscriminator>;
template class TauDiscriminationProducerBase<CaloTau, CaloTauDiscriminator>;
