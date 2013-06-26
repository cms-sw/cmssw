// -*- C++ -*-
//
// Package:    HcalRecHitReflagger
// Class:      HcalRecHitReflagger
// 
/**\class HcalRecHitReflagger HcalRecHitReflagger.cc 

 Description: [Creates a new collection of rechits from an existing collection, and changes rechit flags according to new set of user-supplied conditions.  ]

*/
//
// Original Author:  Dinko Ferencek,8 R-004,+41227676479,  Jeff Temple, 6-1-027
//         Created:  Thu Mar 11 13:42:11 CET 2010
// $Id: HcalRecHitReflagger.cc,v 1.8 2013/02/28 21:06:50 chrjones Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"


using namespace std;
using namespace edm;

//
// class declaration
//

class HcalRecHitReflagger : public edm::EDProducer {
   public:
      explicit HcalRecHitReflagger(const edm::ParameterSet&);
      ~HcalRecHitReflagger();

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginRun(const Run& r, const EventSetup& c) override;

  // Threshold function gets values from polynomial-parameterized functions
  double GetThreshold(const int base, const std::vector<double>& params);
  double GetThreshold(const double base, const std::vector<double>& params);

  // Perform a check of a rechit's S9/S1 value compared to a given threshold
  bool   CheckS9S1(const HFRecHit& hf); 

  // Perform a check of a rechit's |L-S|/|L+S| ratio, compared to a paremeterized threshold
  bool   CheckPET(const HFRecHit& hf);

  // Get S9S1, PET values
  double GetS9S1value(const HFRecHit& hf);
  double GetPETvalue(const HFRecHit& hf);
  double GetSlope(const int ieta, const std::vector<double>& params); 

  // ----------member data ---------------------------
  edm::InputTag hfInputLabel_;
  int  hfFlagBit_;

  // Select the test you wish to run
  bool hfBitAlwaysOn_;
  bool hfBitAlwaysOff_;
  bool hf_Algo3test_;
  bool hf_Algo2test_;
  bool hf_Algo3_AND_Algo2_;
  bool hf_Algo3_OR_Algo2_;

  std::vector<double> S9S1_EnergyThreshLong_;
  std::vector<double> S9S1_ETThreshLong_;
  std::vector<double> S9S1_EnergyThreshShort_;
  std::vector<double> S9S1_ETThreshShort_;
  std::vector<double> S9S1_optimumslope_;
  std::vector<double> PET_EnergyThreshLong_;
  std::vector<double> PET_ETThreshLong_;
  std::vector<double> PET_EnergyThreshShort_;
  std::vector<double> PET_ETThreshShort_;
  std::vector<double> PET_ratiocut_;
  int debug_;

  // Store channels marked as bad in the channel status map
  std::map<HcalDetId, unsigned int> badstatusmap;

  edm::Handle<HFRecHitCollection> hfRecHits;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HcalRecHitReflagger::HcalRecHitReflagger(const edm::ParameterSet& ps)
{
   //register your products
   produces<HFRecHitCollection>();

   hfInputLabel_        = ps.getUntrackedParameter<InputTag>("hfInputLabel",edm::InputTag("hfreco"));
   hfFlagBit_           = ps.getUntrackedParameter<int>("hfFlagBit",HcalCaloFlagLabels::UserDefinedBit0); 
   debug_               = ps.getUntrackedParameter<int>("debug",0);
   
   // sanity check bits -- turn bit on or off always
   hfBitAlwaysOn_       = ps.getUntrackedParameter<bool>("hfBitAlwaysOn",false);
   hfBitAlwaysOff_      = ps.getUntrackedParameter<bool>("hfBitAlwaysOff",false);

   // Enable/disable various tests
   hf_Algo3test_         = ps.getUntrackedParameter<bool>("hf_Algo3test",false);
   hf_Algo2test_          = ps.getUntrackedParameter<bool>("hf_Algo2test",false);
   hf_Algo3_AND_Algo2_     = ps.getUntrackedParameter<bool>("hf_Algo3_AND_Algo2",false);
   hf_Algo3_OR_Algo2_      = ps.getUntrackedParameter<bool>("hf_Algo3_OR_Algo2",false);

   // Get thresholds for each test
   const edm::ParameterSet& S9S1   = ps.getParameter<edm::ParameterSet>("hf_S9S1_params");
   S9S1_EnergyThreshLong_              = S9S1.getUntrackedParameter<std::vector<double> >("S9S1_EnergyThreshLong"); 
   S9S1_ETThreshLong_                  = S9S1.getUntrackedParameter<std::vector<double> >("S9S1_ETThreshLong"); 
   S9S1_EnergyThreshShort_              = S9S1.getUntrackedParameter<std::vector<double> >("S9S1_EnergyThreshShort"); 
   S9S1_ETThreshShort_                  = S9S1.getUntrackedParameter<std::vector<double> >("S9S1_ETThreshShort"); 
   S9S1_optimumslope_                  = S9S1.getParameter<std::vector<double> >("S9S1_optimumslope");

   const edm::ParameterSet& PET    = ps.getParameter<edm::ParameterSet>("hf_PET_params");
   PET_EnergyThreshLong_               = PET.getUntrackedParameter<std::vector<double> >("PET_EnergyThreshLong"); 
   PET_ETThreshLong_                   = PET.getUntrackedParameter<std::vector<double> >("PET_ETThreshLong"); 
   PET_EnergyThreshShort_               = PET.getUntrackedParameter<std::vector<double> >("PET_EnergyThreshShort"); 
   PET_ETThreshShort_                   = PET.getUntrackedParameter<std::vector<double> >("PET_ETThreshShort"); 
   PET_ratiocut_                   = PET.getParameter<std::vector<double> >("PET_ratiocut");
}


HcalRecHitReflagger::~HcalRecHitReflagger()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void HcalRecHitReflagger::beginRun(const Run& r, const EventSetup& c)
{
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  const HcalChannelQuality& chanquality_(*p.product());

  std::vector<DetId> mydetids = chanquality_.getAllChannels();
  for (std::vector<DetId>::const_iterator i = mydetids.begin();i!=mydetids.end();++i)
    {
      if (i->det()!=DetId::Hcal) continue; // not an hcal cell
      HcalDetId id=HcalDetId(*i);
      int status=(chanquality_.getValues(id))->getValue();
      if ( (status & (1<<HcalChannelStatus::HcalCellDead))==0 ) continue;
      badstatusmap[id]=status;
    }

}

// ------------ method called to produce the data  ------------
void
HcalRecHitReflagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // read HF RecHits
   if (!iEvent.getByLabel(hfInputLabel_,hfRecHits))
     {
       if (debug_>0) std::cout <<"Unable to find HFRecHitCollection with label '"<<hfInputLabel_<<"'"<<std::endl;
       return;
     }

   // prepare the output HF RecHit collection
   std::auto_ptr<HFRecHitCollection> pOut(new HFRecHitCollection());
   
   // loop over rechits, and set the new bit you wish to use
   for (HFRecHitCollection::const_iterator recHit=hfRecHits->begin(); recHit!=hfRecHits->end(); ++recHit) {

     HFRecHit newhit = (HFRecHit)(*recHit);
     // Set bit to be on for all hits
     if (hfBitAlwaysOn_)
       newhit.setFlagField(1,hfFlagBit_);

     // Set bit to be off for all hits
     else if (hfBitAlwaysOff_)
       newhit.setFlagField(0,hfFlagBit_);

     else
       {
	 // Check booleans -- true booleans mean that the channel seems noisy
	 bool Algo2bool = false;  // Algorithm 2 is PET testing for both long & short
	 bool Algo3bool = false;  // Algorithm 3 is PET testing for short, S9S1 for long

	 HcalDetId id(newhit.detid().rawId());
	 int depth = newhit.id().depth();
	 int ieta  = newhit.id().ieta();

	 // Code below performs Algo2 or Algo3 tests;  modify as you see fit
	 
	 if (depth==2)  // short fibers -- use PET test everywhere
	   {
	     Algo2bool   = CheckPET(newhit);
	     Algo3bool   = Algo2bool;
	   }
	 else if (depth==1) // long fibers
	   {
	     Algo2bool   = CheckPET(newhit);
	     abs(ieta) == 29 ?  Algo3bool = Algo2bool : Algo3bool   = CheckS9S1(newhit);
	   }

	 int flagvalue=-1;  // new value for flag  (-1 means flag not changed)

	 // run through test combinations
	 if (hf_Algo3_AND_Algo2_==true)
	   (Algo2bool && Algo3bool)==true ? flagvalue=1 : flagvalue = 0;
	 else if (hf_Algo3_OR_Algo2_==true)
	   (Algo2bool || Algo3bool)==true ? flagvalue=1 : flagvalue = 0;
	 else if (hf_Algo3test_ ==true)
	     Algo3bool==true ? flagvalue = 1 : flagvalue = 0;
	 else if (hf_Algo2test_ ==true)
	   Algo2bool==true ? flagvalue = 1: flagvalue = 0;

	 // Set flag bit based on test; if no tests set, don't change flag
	 if (flagvalue!=-1)
	   newhit.setFlagField(flagvalue, hfFlagBit_);
       } // hfBitAlwaysOn/Off bits not set; run other tests

     // Add rechit to collection
     pOut->push_back(newhit);
   }
   
   // put the re-flagged HF RecHit collection into the Event
   iEvent.put(pOut);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalRecHitReflagger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalRecHitReflagger::endJob() {
}

bool HcalRecHitReflagger::CheckPET(const HFRecHit& hf)
{
  /*
    Runs 'PET' test:  Computes ratio |L-S|/(L+S) for channel passing ieta-dependent energy and ET cuts.
    Channel is marked as noise if ratio exceeds ratio R(E)
  */

  HcalDetId id(hf.detid().rawId());
  int iphi = id.iphi();
  int ieta = id.ieta();
  double energy=hf.energy();
  int depth = id.depth();
  double fEta=fabs(0.5*(theHFEtaBounds[abs(ieta)-29]+theHFEtaBounds[abs(ieta)-28]));
  double ET = energy/cosh(fEta);
  double threshold=0;
  
  if (debug_>0) std::cout <<"<RechitReflagger::CheckPET>  energy = "<<energy<<"  ieta = "<<ieta<<std::endl;

  if (depth==1)  //long fibers
    {
      // Check that energy, ET above threshold
      threshold=GetThreshold(ieta, PET_EnergyThreshLong_);
      if (energy<threshold)
	return false;
      threshold=GetThreshold(ieta, PET_ETThreshLong_);
      if (ET<threshold)
	return false;
      
      // Get threshold cut to use at this energy, and compute the |L-S|/(L+S) ratio

      double PETcut=GetThreshold(energy,PET_ratiocut_);
      double PETvalue=GetPETvalue(hf);
      if (debug_>1)
	std::cout <<"  <HcalRecHitReflagger::CheckPET>  ieta = "<<ieta<<"  energy threshold = "<<GetThreshold(ieta, PET_EnergyThreshLong_)<<"  ET threshold = "<<GetThreshold(ieta, PET_ETThreshLong_)<<"   ENERGY = "<<energy<<"   PET threshold = "<<PETcut<<std::endl;

      if (PETvalue>PETcut) 
	{
	  if (debug_>2) std::cout <<"***** FOUND CHANNEL FAILING PET CUT!  CHANNEL = "<<ieta<<",  "<<iphi<<",  "<<depth<<std::endl;
	  return true;  // channel looks noisy
	}
      else return false; // channel doesn't look noisy
    } // if (depth==1)

  else if (depth==2)  //short fibers
    {
      threshold=GetThreshold(ieta, PET_EnergyThreshShort_);
      if (energy<threshold)
	return false;
      threshold=GetThreshold(ieta, PET_ETThreshShort_);
      if (ET<threshold)
	return false;
      
      double PETcut=GetThreshold(energy,PET_ratiocut_);
      double PETvalue=GetPETvalue(hf);
      if (debug_>1)
	std::cout <<"  <HcalRecHitReflagger::CheckPET>  ieta = "<<ieta<<"  energy threshold = "<<GetThreshold(ieta, PET_EnergyThreshShort_)<<"  ET threshold = "<<GetThreshold(ieta, PET_ETThreshShort_)<<"   ENERGY = "<<energy<<"   PET threshold = "<<PETcut<<endl;

      if (PETvalue>PETcut) 
	{
	  if (debug_>2) std::cout <<"***** FOUND CHANNEL FAILING PET CUT!  CHANNEL = "<<ieta<<",  "<<iphi<<",  "<<depth<<std::endl;
	
	  return true; // looks noisy
	} 
      else return false; //doesn't look noisy
    } // else if (depth==2)
  return false; // should never reach this part
} // CheckPET(HFRecHit& hf)



bool HcalRecHitReflagger::CheckS9S1(const HFRecHit& hf)
{
  // Apply S9/S1 test  -- only sensible for long fibers at the moment!

  HcalDetId id(hf.detid().rawId());
  int iphi = id.iphi();
  int ieta = id.ieta();
  double energy=hf.energy();
  int depth = id.depth();
  double fEta=fabs(0.5*(theHFEtaBounds[abs(ieta)-29]+theHFEtaBounds[abs(ieta)-28]));
  double ET = energy/cosh(fEta);
  double threshold=0;

  //double S9S1slope=GetSlope(ieta,S9S1_optimumslope_ ); // get optimized slope for this ieta
  double S9S1slope=S9S1_optimumslope_[abs(ieta)-29];
  double S9S1value=GetS9S1value(hf); // get the ratio S9/S1

  if (debug_>0) std::cout <<"<RechitReflagger::CheckS9S1>  energy = "<<energy<<"  ieta = "<<ieta<<"  S9S1slope = "<<S9S1slope<<"  S9S1value = "<<S9S1value<<std::endl;


  if (depth==1) // long fiber
    {
      // Step 1:  Check that energy above threshold
      threshold=GetThreshold(ieta, S9S1_EnergyThreshLong_);
      if (energy<threshold)
	return false;
      threshold=GetThreshold(ieta, S9S1_ETThreshLong_);
      if (ET<threshold)
	return false;

      // S9S1 cut has form [0]+[1]*log[E], 
      double S9S1cut = -1.*S9S1slope*log(GetThreshold(ieta, PET_EnergyThreshLong_))+S9S1slope*(log(energy));

      if (S9S1value<S9S1cut) 
	{
	  if (debug_>0) std::cout <<"***** FOUND CHANNEL FAILING S9S1 CUT!  CHANNEL = "<<ieta<<",  "<<iphi<<",  "<<depth<<std::endl;
	  return true;
	}
      else return false;
    }

  // ****************WARNING!!!!!****************  SHORT FIBERS DON'T HAVE TUNED THRESHOLDS!  USE PET CUT FOR SHORT FIBERS! ******* //
  else if (depth==2)  //short fiber
    {
      // Step 1:  Check that energy above threshold
      threshold=GetThreshold(ieta, S9S1_EnergyThreshShort_);
      if (energy<threshold)
	return false;
      threshold=GetThreshold(ieta, S9S1_ETThreshShort_);
      if (ET<threshold)
	return false;

      // S9S1 cut has form [0]+[1]*log[E], 
      double S9S1cut = -1.*S9S1slope*log(GetThreshold(ieta, PET_EnergyThreshShort_))+S9S1slope*(log(energy));
   
      if (S9S1value<S9S1cut) 
	{
	  if (debug_>0) std::cout <<"***** FOUND CHANNEL FAILING S9S1 CUT!  CHANNEL = "<<ieta<<",  "<<iphi<<",  "<<depth<<std::endl;
	  return true;
	}
      else return false;
    }
  return false;
}




double HcalRecHitReflagger::GetS9S1value(const HFRecHit& hf)
{
  // sum all energies around cell hf (including hf itself)
  // Then subtract hf energy, and divide by this energy to form ratio S9/S1

  HcalDetId id(hf.detid().rawId());
  int iphi = id.iphi();
  int ieta = id.ieta();
  double energy=hf.energy();
  int depth = id.depth();
  if (debug_>0)  std::cout <<"\t<HcalRecHitReflagger::GetS9S1value>  Channel = ("<<iphi<<",  "<<ieta<<",  "<<depth<<")  :  Energy = "<<energy<<std::endl; 

  double S9S1=0; // starting value

  int testphi=-99;
 
  // Part A:  Check fixed iphi, and vary ieta
  for (int d=1;d<=2;++d) // depth loop
    {
      for (int i=ieta-1;i<=ieta+1;++i) // ieta loop
	{
	  testphi=iphi;
	  // Special case when ieta=39, since ieta=40 only has phi values at 3,7,11,...
	  // phi=3 covers 3,4,5,6
	  if (abs(ieta)==39 && abs(i)>39 && testphi%4==1)
	    testphi-=2;
	  while (testphi<0) testphi+=72;
	  if (i==ieta && d==depth) continue;  // don't add the cell itself
	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, i,testphi,d);
	  HFRecHitCollection::const_iterator neigh=hfRecHits->find(neighbor);
	  if (neigh!=hfRecHits->end())
	    S9S1+=neigh->energy();
	}
    }

  // Part B: Fix ieta, and loop over iphi.  A bit more tricky, because of iphi wraparound and different segmentation at 40, 41
  
  int phiseg=2; // 10 degree segmentation for most of HF (1 iphi unit = 5 degrees)
  if (abs(ieta)>39) phiseg=4; // 20 degree segmentation for |ieta|>39
  for (int d=1;d<=2;++d)
    {
      for (int i=iphi-phiseg;i<=iphi+phiseg;i+=phiseg)
	{
	  if (i==iphi) continue;  // don't add the cell itself, or its depthwise partner (which is already counted above)
	  testphi=i;
	  // Our own modular function, since default produces results -1%72 = -1
	  while (testphi<0) testphi+=72;
	  while (testphi>72) testphi-=72;
	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, ieta,testphi,d);
	  HFRecHitCollection::const_iterator neigh=hfRecHits->find(neighbor);
	  if (neigh!=hfRecHits->end())
	    S9S1+=neigh->energy();
	}
    }
  
  if (abs(ieta)==40) // add extra cells for 39/40 boundary due to increased phi size at ieta=40.
    {
      for (int d=1;d<=2;++d) // add cells from both depths!
	{
	  HcalDetId neighbor(HcalForward, 39*abs(ieta)/ieta,(iphi+2)%72,d);  
	  HFRecHitCollection::const_iterator neigh=hfRecHits->find(neighbor);
	  if (neigh!=hfRecHits->end())
	      S9S1+=neigh->energy();
	}
    }
    
  S9S1/=energy;  // divide to form actual ratio
  return S9S1;
} // GetS9S1value


double HcalRecHitReflagger::GetPETvalue(const HFRecHit& hf)
{
  HcalDetId id(hf.detid().rawId());
  int iphi = id.iphi();
  int ieta = id.ieta();
  int depth = id.depth();
  double energy=hf.energy();
  if (debug_>0)  std::cout <<"\t<HcalRecHitReflagger::GetPETvalue>  Channel = ("<<iphi<<",  "<<ieta<<",  "<<depth<<")  :  Energy = "<<energy<<std::endl; 

  HcalDetId pId(HcalForward, ieta,iphi,3-depth); // get partner;
  // Check if partner is in known dead cell list; if so, don't flag
  if (badstatusmap.find(pId)!=badstatusmap.end())
    return 0;

  double epartner=0; // assume no partner
  // TO DO:  Add protection against cells marked dead in status database
  HFRecHitCollection::const_iterator part=hfRecHits->find(pId);
  if ( part!=hfRecHits->end() ) epartner=part->energy();
  return 1.*(energy-epartner)/(energy+epartner);
}


double HcalRecHitReflagger::GetThreshold(const int base, const std::vector<double>& params)
{
  double thresh=0;
  for (unsigned int i=0;i<params.size();++i)
    {
      thresh+=params[i]*pow(fabs(base),(int)i);
    }
  return thresh;
}

double HcalRecHitReflagger::GetThreshold(const double base, const std::vector<double>& params)
{
  double thresh=0;
  for (unsigned int i=0;i<params.size();++i)
    {
      thresh+=params[i]*pow(fabs(base),(int)i);
    }
  return thresh;
}

double HcalRecHitReflagger::GetSlope(const int ieta, const std::vector<double>& params)
{
  double slope=0;
  if (abs(ieta)==40 && params.size()>0)
    slope= params[0];
  else if (abs(ieta)==41 && params.size()>1)
    slope= params[1];
  else if (params.size()>2)
    {
      for (unsigned int i=2;i<params.size();++i)
	slope+=params[i]*pow(static_cast<double>(abs(ieta)),static_cast<int>(i)-2);
    }
  if (debug_>1)  std::cout <<"<HcalRecHitReflagger::GetSlope>  ieta = "<<ieta<<"  slope = "<<slope<<std::endl;
  return slope;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalRecHitReflagger);
