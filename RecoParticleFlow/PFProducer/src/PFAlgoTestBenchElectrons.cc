// Dummy Test For Electrons Daniele.Benedetti@cern.ch


#include "RecoParticleFlow/PFProducer/interface/PFAlgoTestBenchElectrons.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

using namespace std;
using namespace reco;

 
void PFAlgoTestBenchElectrons::processBlock(const reco::PFBlockRef& blockref,
					    std::list<PFBlockRef>& hcalBlockRefs,
					    std::list<PFBlockRef>& ecalBlockRefs)
{

  //cout<<"electron test bench: process block"
  //    <<(*blockref)<<endl;
 

  const reco::PFBlock& block = *blockref;
  const PFBlock::LinkData& linkData =  block.linkData();
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();

  std::vector<bool> active(elements.size(), true );
  vector<unsigned> mainecal_index;
  for(unsigned iEle=0; iEle<elements.size(); iEle++) {
    PFBlockElement::Type type = elements[iEle].type();
    if (type == reco::PFBlockElement::ECAL ) {
      mainecal_index.push_back(iEle);
    }
  }
  
  std::map<unsigned, unsigned> FinalBremECALIndex;
  unsigned ECALGSF_index = 1000;
  unsigned GSF_index = 1000;
  for(unsigned iEle=0; iEle<elements.size(); iEle++) {
    
    
    PFBlockElement::Type type = elements[iEle].type();
    
    if (type == reco::PFBlockElement::GSF ) {
      GSF_index = iEle;
      
      // linked ECAL clusters
      std::multimap<double, unsigned> ecalElems;
      block.associatedElements( iEle,  linkData,
				  ecalElems ,
				  reco::PFBlockElement::ECAL );     
      
      typedef std::multimap<double, unsigned>::iterator IE;
      
      float chi2GSFFinal = 1000;
      for(IE ie = ecalElems.begin(); ie != ecalElems.end(); ++ie ) {
	double   chi2  = ie->first;
	unsigned index = ie->second;
	if (chi2 <  chi2GSFFinal) {
	  chi2GSFFinal = chi2;
	  ECALGSF_index = index;
	}
      }
      
      std::multimap<double, unsigned> Brems;
      
      block.associatedElements( iEle,  linkData,
				  Brems ,
				  reco::PFBlockElement::BREM );
      typedef std::multimap<double, unsigned>::iterator IB;
	   
      vector< pair<double, unsigned> > ecalindexforbrem;
      vector<unsigned> brem_index;
      for(IB iee = Brems.begin(); iee != Brems.end(); ++iee ) {

	unsigned index = iee->second;

	std::multimap<double, unsigned> ecalBrems;
	block.associatedElements( index,  linkData,
				    ecalBrems ,
				    reco::PFBlockElement::ECAL );
	typedef std::multimap<double, unsigned>::iterator IEB;
	float chi2BremFinal = 1000;
	unsigned ECALBREM_index = 1000;
	
	for(IEB ieb = ecalBrems.begin(); ieb != ecalBrems.end(); ++ieb ) {
	  // Two cases: first there is a ecalGSF=brem chi2 < chi2ecalGSF ?
	  
	  
	  if (ieb->second != ECALGSF_index) {
	    
	    double   chi2_eb  = ieb->first;
	    unsigned index_eb = ieb->second;
	    if (chi2_eb < chi2BremFinal) {
	      chi2BremFinal = chi2_eb;
	      ECALBREM_index = index_eb;
	    }
	  }
	}
	if (ECALBREM_index != 1000){
	  ecalindexforbrem.push_back(make_pair(chi2BremFinal,ECALBREM_index));  
	  // for each brem (same size that brems)
	  brem_index.push_back(index);
	}
      }
      for (unsigned ie=0;ie<mainecal_index.size();ie++){
	unsigned c_brem = 0;
	
	double chi2best = 1000;
	unsigned final_ecal = 1000;
	unsigned final_brem = 1000;
	if (mainecal_index[ie] != ECALGSF_index) {
	  for (unsigned i2eb = 0; i2eb < ecalindexforbrem.size(); i2eb++ ) {
	    unsigned temp_ecal_index =  (ecalindexforbrem[i2eb]).second;
	    double temp_chi2 = (ecalindexforbrem[i2eb]).first;

	    if (temp_ecal_index == mainecal_index[ie] ) {
	      if (temp_chi2 < chi2best ){
		chi2best = temp_chi2;
		final_ecal = temp_ecal_index;
		final_brem = brem_index[c_brem];
	      }
	    }
	    c_brem++;
	  }
	  if (chi2best < 1000) {
	    FinalBremECALIndex.insert(make_pair(final_brem,final_ecal)); 
	  }
	}
	
      }


      
      bool ElectronId = false;
     
      // perform electron iD
      //Start  ***************************************
      ElectronId = true; 
      //End ***************************************
      
      // Lock the various elements 
      // NOTE very lose cuts!
      if (ElectronId) {
	if (chi2GSFFinal < 1000) {      

	  active[iEle] = false;   //GSF 
	  active[ECALGSF_index] = false;  // ECAL associated to GSF
	  // Lock Brem and ECAL associated to brem
	  typedef std::map<unsigned, unsigned>::iterator IT;
	  for (map<unsigned, unsigned>::iterator it = FinalBremECALIndex.begin(); it != FinalBremECALIndex.end(); ++it ) {
	    unsigned index_brem = it->first;
	    unsigned index_ecalbrem = it->second;
	    active[index_brem]=false;
	    active[index_ecalbrem] = false;
	  }
	}
      }  // End if Electron ID
    } // End GSF type
  } // Run on elements
   

  vector<pair< bool, vector<unsigned int> > > associate;

  for(unsigned iEle=0; iEle<elements.size(); iEle++) {

    if (active[iEle] == true) {
      vector<unsigned> assoel(0);
      associate.push_back(make_pair(active[iEle],assoel));
    }
    if (active[iEle] == false){
      PFBlockElement::Type type = elements[iEle].type();
      if (type == reco::PFBlockElement::GSF) {
	vector<unsigned> assoel(0);
	// push back tk track: to be done
	
	// push back ecal cluster associate to gsf track
	assoel.push_back(ECALGSF_index);
	// push back selected brems
	for (map<unsigned, unsigned>::iterator it = FinalBremECALIndex.begin(); it != FinalBremECALIndex.end(); ++it ) {
	  unsigned index_brem = it->first;
	  assoel.push_back(index_brem);
	}
	associate.push_back(make_pair(active[iEle],assoel));
	// push back hcal cluster to be done
	// push back ps cluster to be done
      }
      if (type == reco::PFBlockElement::BREM) {
	vector<unsigned> assoel(0);
	// push back GSF track
	assoel.push_back(GSF_index);
	
	// push_back ecal clusters associate to the brem
	for (map<unsigned, unsigned>::iterator it = FinalBremECALIndex.begin(); it != FinalBremECALIndex.end(); ++it ) {
	  if (it->first == iEle) {
	    unsigned index_ecalbrem = it->second;
	    assoel.push_back(index_ecalbrem);
	  }
	}
	associate.push_back(make_pair(active[iEle],assoel));
	
	// push back hcal cluster to be done
	// push back ps cluster to be done
      }
      if  (type == reco::PFBlockElement::ECAL) {
	if (iEle == ECALGSF_index) {
	  // cluster associate to the gsf track
	  // push back the associate gsf track
	  vector<unsigned> assoel(0);
	  assoel.push_back(GSF_index);
	  associate.push_back(make_pair(active[iEle],assoel));
	}
	for (map<unsigned, unsigned>::iterator it = FinalBremECALIndex.begin(); it != FinalBremECALIndex.end(); ++it ) {
	  unsigned index_ecalbrem = it->second;
	  unsigned index_brem = it->first;
	  if (iEle == index_ecalbrem) {    // push back the brems associate to ecal 
	    vector<unsigned> assoel(0);
	    assoel.push_back(index_brem);
	    associate.push_back(make_pair(active[iEle],assoel));
	  }
	}
      }
    }
  }
  // here create the object with the block and associate (and active if needed)

} // End loop on Blocks


