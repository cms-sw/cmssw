#include "HLTriggerOffline/Tau/interface/HLTTauRefCombiner.h"
#include "Math/GenVector/VectorUtil.h"



using namespace edm;
using namespace std;

HLTTauRefCombiner::HLTTauRefCombiner(const edm::ParameterSet& iConfig)
{
  std::vector<edm::InputTag> inputCollVector_ = iConfig.getParameter< std::vector<InputTag> >("InputCollections");
  for(unsigned int ii=0; ii<inputCollVector_.size(); ++ii)
    {
      inputColl_.push_back( consumes<LorentzVectorCollection>(inputCollVector_[ii]) );
    }
  matchDeltaR_ = iConfig.getParameter<double>("MatchDeltaR");
  outName_     = iConfig.getParameter<string>("OutputCollection");

  produces<LorentzVectorCollection> ( outName_);
}

HLTTauRefCombiner::~HLTTauRefCombiner(){ }

void HLTTauRefCombiner::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
    unique_ptr<LorentzVectorCollection> out_product(new LorentzVectorCollection);

    //Create The Handles..
    std::vector< Handle<LorentzVectorCollection> > handles;

    bool allCollectionsExist = true;
    //Map the Handles to the collections if all collections exist
    for(size_t i = 0;i<inputColl_.size();++i)
      {
	edm::Handle<LorentzVectorCollection> tmp;
	if(iEvent.getByToken(inputColl_[i],tmp))
	  {
	    handles.push_back(tmp);
	  }
	else
	  {
	    allCollectionsExist = false;
	  }

      }

    //The reference output object collection will be the first one..
    if(allCollectionsExist)
      {
	//loop on the first collection
	for(size_t i = 0; i < (handles[0])->size();++i)
	  {
	    bool MatchedObj = true;

	    //get reference Vector
	    const LorentzVector lvRef = (*(handles[0]))[i];
	    
	    //Loop on all other collections and match
	    	for(size_t j = 1; j < handles.size();++j)
		  {
		    if(!match(lvRef,*(handles[j])))
		      MatchedObj = false;

		  }
		
		//If the Object is Matched Everywhere store
		if(MatchedObj)
		  {
		    out_product->push_back(lvRef);
		  }


	  }

	//Put product to file
	iEvent.put(std::move(out_product),outName_);

      }
    
    


}



bool 
HLTTauRefCombiner::match(const LorentzVector& lv,const LorentzVectorCollection& lvcol)
{
 bool matched=false;

 if(!lvcol.empty())
  for(LorentzVectorCollection::const_iterator it = lvcol.begin();it!=lvcol.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(lv,*it);
	  if(delta<matchDeltaR_)
	    {
	      matched=true;
	     
	    }
   }



 return matched;
}
