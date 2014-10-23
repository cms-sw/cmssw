#ifndef _StringCutEventSelector
#define _StringCutEventSelector

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "CommonTools/UtilAlgos/interface/InputTagDistributor.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

template<typename Object, bool any=false>
class  StringCutEventSelector : public EventSelector {
 public:
  StringCutEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) :
    StringCutEventSelector(pset, iC) {}
  StringCutEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) :
    EventSelector(pset, iC),
    src_(edm::Service<InputTagDistributorService>()->retrieve("src",pset)),
    srcToken_(iC.consumes<edm::View<Object> >(src_)),
    f_(pset.getParameter<std::string>("cut")),
    //put this guy to 0 to do the check on "all" object in the collection
    nFirst_(pset.getParameter<unsigned int>("nFirst")),
    order_(0)
      {
	  std::stringstream ss;
	  ss<<"string cut based selection on collection: "<<src_;
	  description_.push_back(ss.str());
	  ss.str("");
	  description_.push_back(std::string("selection cut is: ")+pset.getParameter<std::string>("cut"));
	  if (pset.exists("order"))
	    order_ = new StringObjectFunction<Object>( pset.getParameter<std::string>("order"));
      }

    bool select (const edm::Event& e) const{
      edm::Handle<edm::View<Object> > oH;
      e.getByToken(srcToken_, oH);
      std::vector<const Object*> copyToSort(oH->size());
      for (uint i=0;i!=oH->size();++i)  copyToSort[i]= &(*oH)[i];
      if (order_)std::sort(copyToSort.begin(), copyToSort.end(), sortByStringFunction<Object>(order_));
      
      //reject events if not enough object in collection
      //      if ((nFirst_!=0) && (oH->size()<nFirst_)) return false;
      unsigned int i=0;
      unsigned int found=0;
      for (;i!=oH->size();i++)
	{ 
	  const Object & o = *(copyToSort[i]);
	  if (any){
	    if (f_(o)) ++found;
	    if (found>=nFirst_)
	      return true;
	  }
	  else{
	    //stop doing the check if reaching too far in the collection
	    if ((nFirst_!=0) && (i>=nFirst_)) break;
	    if (!f_(o)) return false;
	  }
	}
      return !any;
    }

 private:
    edm::InputTag src_;
    edm::EDGetTokenT<edm::View<Object> > srcToken_;
    StringCutObjectSelector<Object> f_;
    unsigned int nFirst_;
    StringObjectFunction<Object> *order_;
};


template<typename Object, bool existenceMatter=true>
class  StringCutsEventSelector : public EventSelector {
 public:
  StringCutsEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) :
    StringCutsEventSelector(pset, iC) {}
  StringCutsEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) :
    EventSelector(pset, iC),
    src_(edm::Service<InputTagDistributorService>()->retrieve("src",pset)),
    srcToken_(iC.consumes<edm::View<Object> >(src_)),
    order_(0)
      {
	std::vector<std::string> selection=pset.getParameter<std::vector<std::string > >("cut");
	std::stringstream ss;
	ss<<"string cut based selection on collection: "<<src_;
	description_.push_back(ss.str());	    ss.str("");
	description_.push_back("selection cuts are:");
	for (unsigned int i=0;i!=selection.size();i++)
	  if (selection[i]!="-"){
	    f_.push_back( new StringCutObjectSelector<Object>(selection[i]));
	    ss<<"["<<i<<"]: "<<selection[i];
	    description_.push_back(ss.str());           ss.str("");
	  }
	  else
	    {
	      f_.push_back(0);
	      ss<<"["<<i<<"]: no selection";
	      description_.push_back(ss.str());           ss.str("");
	    }
	if (pset.exists("order"))
	  order_ = new StringObjectFunction<Object>( pset.getParameter<std::string>("order"));
      }
  ~StringCutsEventSelector(){
    unsigned int i=0; 
    for (;i!=f_.size();i++) 
      if (f_[i]){ 
	delete f_[i];f_[i]=0;
      }
    if (order_) delete order_;
  }

    bool select (const edm::Event& e) const{
      edm::Handle<edm::View<Object> > oH;
      e.getByToken(srcToken_, oH);
      std::vector<const Object*> copyToSort(oH->size());
      for (uint i=0;i!=oH->size();++i)  copyToSort[i]= &(*oH)[i];
      if (order_)std::sort(copyToSort.begin(), copyToSort.end(), sortByStringFunction<Object>(order_));

      unsigned int i=0;
      if (existenceMatter && oH->size()<f_.size()) return false;
      for (;i!=f_.size();i++)
	{
	  if (!existenceMatter && i==oH->size()) break;
	  if (!f_[i]) continue;
	  const Object & o = *(copyToSort[i]);
	  if (!(*f_[i])(o)) return false;
	}
      return true;
    }

 private:
    edm::InputTag src_;
    edm::EDGetTokenT<edm::View<Object> > srcToken_;
    std::vector<StringCutObjectSelector<Object> *> f_;
    StringObjectFunction<Object> * order_;
};

#endif
