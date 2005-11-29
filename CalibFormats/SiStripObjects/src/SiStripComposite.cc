#include "CalibTracker/SiStripConnectivity/interface/SiStripComposite.h" 
#include <sstream>
using namespace std;
//
// -- Constructors
//

SiStripComposite::SiStripComposite(){
  theName = "";
  theHeader = 0;
  theModuleConnection =0; 
}

SiStripComposite::SiStripComposite(ModuleConnection v, 
				 string type,
				 string name,
				 string position, 
				 CompositeHeader* h) : theName(name), theType(type),
						       thePosition(position),
						       theHeader(h) {
  this->setModuleConnection(v);
}
SiStripComposite::SiStripComposite(string type,
				 string name,
				 string position, 
				 CompositeHeader* h) : theName(name), theType(type),
						       thePosition(position),
						       theHeader(h),
                                                       theModuleConnection(0) {}

SiStripComposite::SiStripComposite(SiStripCompositeVectorType v, 
				 string type,
				 string name,
				 string position, 
				 CompositeHeader* h) : theName(name), theType(type),
						       thePosition(position),
						       theHeader(h),
                                                       theModuleConnection(0) {
  this->addStructures(v);
}
//
// -- Destructor
//
SiStripComposite::~SiStripComposite() {
   delete theModuleConnection;
   theSiStripCompositeList.clear();
}
//
// -- Add Structure to the Composite
//
void SiStripComposite::addStructure(SiStripComposite* in){
  theSiStripCompositeList.push_back(in);
}
//
// -- Add Structures to the Composite
//
void SiStripComposite::addStructures(SiStripCompositeVectorType in){
  copy(in.begin(), in.end(), back_inserter(theSiStripCompositeList));
}
//
// -- Add ModuleConnection
//
void SiStripComposite::setModuleConnection(ModuleConnection in){
  ModuleConnection* theConnection = new ModuleConnection(in);
  theModuleConnection = theConnection;
  //  cout << " adding Module " << in.getModuleId() << " " 
  //       << this->containsModules() << endl;
}
//
// -- Get end point structures 
//
SiStripComposite::SiStripCompositeVectorType SiStripComposite::deepStructures(){
  //
  // traverse the tree recurvively
  //
  SiStripCompositeVectorType temp;
  if ((this)->structures().size() == 0 ) {
    temp.push_back(this);
    return temp;
  }
  if (containsStructures() == true){
    SiStripCompositeVectorType str = structures();
    for (SiStripCompositeVectorType::iterator it  = str.begin();
	 it != str.end(); it++){
      SiStripCompositeVectorType result = (**it).deepStructures();
      copy (result.begin(), result.end(), back_inserter(temp));
    }
    return temp;
  }
  //  cout <<" SiStripComposite: Does not contain neither modules nor structures...."<<endl;
  return SiStripCompositeVectorType(); 
}
//
// -- Get end point structures 
//
SiStripComposite::ModuleConnectionVectorType SiStripComposite::deepModules(){
  ModuleConnectionVectorType modVec;
  SiStripCompositeVectorType allModules = deepStructures();
  for (SiStripCompositeVectorType::iterator it = allModules.begin(); 
       it != allModules.end(); it ++){
    if ((*it)->containsModules()) 
     modVec.push_back((*it)->getModuleConnection());
  }
  return modVec;
}
//
// -- Get End point module
//
SiStripComposite* SiStripComposite::getModule(string id){
  SiStripCompositeVectorType allModules = deepStructures();
  for (SiStripCompositeVectorType::iterator it = allModules.begin(); 
       it != allModules.end(); it ++){
    if ((*it)->containsModules()) {
      string tk_name = (*it)->name();
      if (!tk_name.compare(id)) {
        return (*it);
      } 
    }
  }
  return 0;
}
//
// -- Get End point module
//
SiStripComposite* SiStripComposite::getModule(int id){
  SiStripCompositeVectorType allModules = deepStructures();
  for (SiStripCompositeVectorType::iterator it = allModules.begin(); 
       it != allModules.end(); it ++){
    if ((*it)->containsModules()) {
      istringstream strin((*it)->name());
      int k;
      strin>>k;
      if (k == id) {
        return (*it);
      } 
    }
  }
  return 0;
}
//
// Print the whole structure
//
void SiStripComposite::print(){
  cout <<" - Composite : "<<this->type()<<" "<<this<<" " 
       <<this->name() <<" Position "<<this->position()<< "  Contains Modules " 
       << this->containsModules()<< endl;
  if (this->containsModules()) theModuleConnection->print();
  if (!this->structures().empty()) cout <<"Going down ... :: contains  "  
	     << this->structures().size() << " structures ..." <<endl;
  for(unsigned int k=0; k<this->structures().size(); k++) {

    ((this->structures())[k])->print();
    cout << "........" << endl;
  }
}
// check if there is a Module Connection
bool SiStripComposite::containsModules(){
  return (theModuleConnection!=0);
}
// get Module Connection
ModuleConnection*  SiStripComposite::getModuleConnection(){
  return theModuleConnection;
}
//
// find a specific Composite Child 
//
SiStripComposite* SiStripComposite::findCompositeChild(string name, string type, string position) {
  for (SiStripComposite::SiStripCompositeVectorType::iterator it = theSiStripCompositeList.begin();
       it != theSiStripCompositeList.end(); it++) {
    if ((*it)->name() == name &&
        (*it)->type() == type &&
        (*it)->position() == position) return (*it);
  }
  return 0;
}
