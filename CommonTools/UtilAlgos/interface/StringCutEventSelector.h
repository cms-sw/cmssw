#ifndef _StringCutEventSelector
#define _StringCutEventSelector

#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "CommonTools/UtilAlgos/interface/InputTagDistributor.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

template<typename Object>
class  StringCutEventSelector : public EventSelector {
 public:
  StringCutEventSelector(const edm::ParameterSet& pset) :
    EventSelector(pset),
    src_(edm::Service<InputTagDistributorService>()->retrieve("src",pset)),
    f_(pset.getParameter<std::string>("cut")),
    //put this guy to 0 to do the check on "all" object in the collection
    nFirst_(pset.getParameter<unsigned int>("nFirst"))
      {
	  std::stringstream ss;
	  ss<<"string cut based selection on collection: "<<src_;
	  description_.push_back(ss.str());
	  ss.str("");
	  description_.push_back(std::string("selection cut is: ")+pset.getParameter<std::string>("cut"));
      }
    
    bool select (const edm::Event& e) const{
      edm::Handle<edm::View<Object> > oH;
      e.getByLabel(src_, oH);
      //reject events if not enough object in collection
      //      if ((nFirst_!=0) && (oH->size()<nFirst_)) return false;
      unsigned int i=0;
      for (;i!=oH->size();i++)
	{
	  //stop doing the check if reaching too far in the collection
	  if ((nFirst_!=0) && (i>=nFirst_)) break;
	  const Object & o = (*oH)[i];
	  if (!f_(o)) return false;
	}
      return true;
    }
    
 private:
    edm::InputTag src_;
    StringCutObjectSelector<Object> f_;
    unsigned int nFirst_;
};


template<typename Object, bool existenceMatter=true>
class  StringCutsEventSelector : public EventSelector {
 public:
  StringCutsEventSelector(const edm::ParameterSet& pset) :
    EventSelector(pset),
    src_(edm::Service<InputTagDistributorService>()->retrieve("src",pset))
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
      }
 ~StringCutsEventSelector(){unsigned int i=0; for (;i!=f_.size();i++) if (f_[i]){ delete f_[i];f_[i]=0;}}
 
    bool select (const edm::Event& e) const{
      edm::Handle<edm::View<Object> > oH;
      e.getByLabel(src_, oH);
      unsigned int i=0;
      if (existenceMatter && oH->size()<f_.size()) return false;
      for (;i!=f_.size();i++)
	{  
	  if (!existenceMatter && i==oH->size()) break;
	  if (!f_[i]) continue;
	  const Object & o = (*oH)[i];
	  if (!(*f_[i])(o)) return false;
	}
      return true;
    }
    
 private:
    edm::InputTag src_;
    std::vector<StringCutObjectSelector<Object> *> f_;
};

#endif
