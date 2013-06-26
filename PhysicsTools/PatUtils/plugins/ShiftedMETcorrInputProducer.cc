#include "PhysicsTools/PatUtils/plugins/ShiftedMETcorrInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

ShiftedMETcorrInputProducer::ShiftedMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  src_ = cfg.getParameter<vInputTag>("src");

//--- check that all InputTags refer to the same module label
//   (i.e. differ by instance label only)
  for ( vInputTag::const_iterator src_ref = src_.begin();
	src_ref != src_.end(); ++src_ref ) {
    for ( vInputTag::const_iterator src_test = src_ref;
	  src_test != src_.end(); ++src_test ) {
      if ( src_test->label() != src_ref->label() )
	throw cms::Exception("ShiftedMETcorrInputProducer") 
	  << "InputTags specified by 'src' Configuration parameter must not refer to different module labels !!\n";
    }
  }

  shiftBy_ = cfg.getParameter<double>("shiftBy");

  if ( cfg.exists("binning") ) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for ( vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin();
	  cfgBinningEntry != cfgBinning.end(); ++cfgBinningEntry ) {
      binning_.push_back(new binningEntryType(*cfgBinningEntry));
    }
  } else {
    double uncertainty = cfg.getParameter<double>("uncertainty");
    binning_.push_back(new binningEntryType(uncertainty));
  }
  
  for ( vInputTag::const_iterator src_i = src_.begin();
	src_i != src_.end(); ++src_i ) {
    for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      produces<CorrMETData>((*binningEntry)->getInstanceLabel_full(src_i->instance()));
    }
  }
}

ShiftedMETcorrInputProducer::~ShiftedMETcorrInputProducer()
{
  for ( std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	it != binning_.end(); ++it ) {
    delete (*it);
  }
}

void ShiftedMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  for ( vInputTag::const_iterator src_i = src_.begin();
	src_i != src_.end(); ++src_i ) {
    for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      edm::Handle<CorrMETData> originalObject;
      evt.getByLabel(edm::InputTag(src_i->label(), (*binningEntry)->getInstanceLabel_full(src_i->instance())), originalObject);
  
      double shift = shiftBy_*(*binningEntry)->binUncertainty_;
      
      std::auto_ptr<CorrMETData> shiftedObject(new CorrMETData(*originalObject));
//--- MET balances momentum of reconstructed particles,
//    hence variations of "unclustered energy" and MET are opposite in sign
      shiftedObject->mex   = -shift*originalObject->mex;
      shiftedObject->mey   = -shift*originalObject->mey;
      shiftedObject->sumet = shift*originalObject->sumet;
      
      evt.put(shiftedObject, (*binningEntry)->getInstanceLabel_full(src_i->instance()));
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedMETcorrInputProducer);
