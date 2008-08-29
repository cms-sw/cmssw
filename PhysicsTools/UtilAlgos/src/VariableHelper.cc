#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"


VariableHelper::VariableHelper(const edm::ParameterSet & iConfig){
  std::vector<std::string> psetNames;
  iConfig.getParameterSetNames(psetNames);
  for (uint i=0;i!=psetNames.size();++i){
    std::string & vname=psetNames[i];
    edm::ParameterSet vPset=iConfig.getParameter<edm::ParameterSet>(psetNames[i]);
    std::string method=vPset.getParameter<std::string>("method");

    std::map<std::string, edm::Entry> indexEntry;
    if (vname.find("_N")!=std::string::npos){
      //will have to loop over indexes
      std::vector<uint> indexes = vPset.getParameter<std::vector<uint> >("indexes");
      for (uint iI=0;iI!=indexes.size();++iI){
	edm::ParameterSet toUse = vPset;
	edm::Entry e("uint",indexes[iI],true);
	std::stringstream ss;
	//add +1 0->1, 1->2, ... in the variable label
	ss<<indexes[iI]+1;
	indexEntry.insert(std::make_pair(ss.str(),e));
      }
    }//contains "_N"

    std::map< std::string, edm::Entry> varEntry;
    if (vname.find("_V")!=std::string::npos){
      //do something fancy for multiple variable from one PSet
      std::vector<std::string> vars = vPset.getParameter<std::vector<std::string> >("vars");
      for (uint v=0;v!=vars.size();++v){
	uint sep=vars[v].find(":");
	std::string name=vars[v].substr(0,sep);
	std::string expr=vars[v].substr(sep+1);
	
	edm::Entry e("string",expr,true);
	varEntry.insert(std::make_pair(name,e));
      }
    }//contains "_V"

    if (indexEntry.empty() && varEntry.empty())
    {
      //std::string type=vPset.getParameter<std::string>("type");
      std::string type="helper";
      if (type=="helper"){
	variables_[vname]=CachingVariableFactory::get()->create(method,vname,vPset);
      }
      else{
	//type not recognized
	throw;
      }
    }//only one variable finished
    else{
      std::string radical = vname;
      //remove the "_V";
      if (!varEntry.empty())
	radical = radical.substr(0,radical.size()-2);
      //remove the "_X";
      if (!indexEntry.empty())
	radical = radical.substr(0,radical.size()-2);

      if(varEntry.empty()){
	//loop only the indexes
	for(std::map< std::string, edm::Entry>::iterator iIt=indexEntry.begin();iIt!=indexEntry.end();++iIt){
	  edm::ParameterSet toUse = vPset;
	  toUse.insert(true,"index",iIt->second);
	  std::string newVname = radical+iIt->first;
	  //	  std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	  variables_[newVname]=CachingVariableFactory::get()->create(method,newVname,toUse);
	}
      }else{
	for (std::map< std::string, edm::Entry>::iterator vIt=varEntry.begin();vIt!=varEntry.end();++vIt){
	  if (indexEntry.empty()){
	    edm::ParameterSet toUse = vPset;
	    toUse.insert(true,"expr",vIt->second);
	    std::string newVname = radical+vIt->first;
	    //	    std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	    variables_[newVname]=CachingVariableFactory::get()->create(method,newVname,toUse);
	  }else{
	    for(std::map< std::string, edm::Entry>::iterator iIt=indexEntry.begin();iIt!=indexEntry.end();++iIt){
	      edm::ParameterSet toUse = vPset;
	      toUse.insert(true,"expr",vIt->second);
	      toUse.insert(true,"index",iIt->second);
	      std::string newVname = radical+iIt->first+vIt->first;
	      //	      std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	      variables_[newVname]=CachingVariableFactory::get()->create(method,newVname,toUse);
	    }}
	}
      }

    }//multiple variable per Pset finished
  }
}

void VariableHelper::setHolder(std::string hn){
  std::map<std::string, CachingVariable*> ::const_iterator it = variables_.begin();
  std::map<std::string, CachingVariable*> ::const_iterator it_end = variables_.end();
  for (;it!=it_end;++it)  it->second->setHolder(hn);
}

void VariableHelper::print(){
  std::map<std::string, CachingVariable*> ::const_iterator it = variables_.begin();
  std::map<std::string, CachingVariable*> ::const_iterator it_end = variables_.end();
  for (;it!=it_end;++it)  it->second->print();
}

/*
void VariableHelper::update(const edm::Event & e, const edm::EventSetup & es) const
{
  ev_=&e;
  es_=&es;
}
*/

const CachingVariable* VariableHelper::variable(std::string name) const{ 
  std::map<std::string, CachingVariable*> ::const_iterator v=variables_.find(name);
  if (v!=variables_.end())
    return v->second;
  else
    {
      edm::LogError("VariableHelper")<<"I don't know anything named: "<<name;
      return 0;
    }
}


/*
double VariableHelper::operator() (std::string & name,const edm::Event & iEvent) const{  
  const CachingVariable* v = variable(name);
  return (*v)();
}

double VariableHelper::operator() (std::string name) const{
  const CachingVariable* v = variable(name);
  return (*v)();
}
*/
