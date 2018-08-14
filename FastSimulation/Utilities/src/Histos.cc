//FAMOS headers
#include "FastSimulation/Utilities/interface/Histos.h"

#include "TFile.h"
//#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include <sstream>

Histos* Histos::myself = nullptr;

Histos::Histos() {}

Histos* Histos::instance() {
  if (!myself) myself = new Histos();
  return myself;
}

Histos::~Histos() {}

void 
Histos::book(const std::string& name, 
	     int nx, float xmin, float xmax,
	     int ny, float ymin, float ymax) {
  
  if ( theHistos.find(name) != theHistos.end() ) { 

    std::cout << "Histos::book() : Histogram " 
	 << name << " exists already. Nothing done" << std::endl;

  } else {

    if ( ny ) {

      theHistos[name] = new TH2F(name.c_str(),"",nx,xmin,xmax,ny,ymin,ymax);
      theTypes[name] = 2;

    } else {

      theHistos[name] = new TH1F(name.c_str(),"",nx,xmin,xmax);
      theTypes[name] = 1;

    }

  }

}

void 
Histos::book(const std::string& name, 
	     int nx, float xmin, float xmax,
	     const std::string& option) {
  
  if ( theHistos.find(name) != theHistos.end() ) { 

    std::cout << "Histos::book() : Histogram " 
	 << name << " exists already. Nothing done" << std::endl;

  } else {

    theHistos[name] = new TProfile(name.c_str(),"",nx,xmin,xmax,option.c_str());
    theTypes[name] = 3;  
  }

}

void 
Histos::put(const std::string& file, std::string name) {

  TFile * f = new TFile(file.c_str(),"recreate");
  f->cd();
  
  HistoItr ho ;
  for(ho=theObjects.begin();ho!=theObjects.end();++ho)
    {
      (*ho).second->Write((*ho).first.c_str());
    }


  HistoItr hh = theHistos.find(name);
  if ( name == "" ) 
    for ( hh  = theHistos.begin(); 
	  hh != theHistos.end(); 
	  ++hh ) { 
      if ( theTypes[(*hh).first] == 1 )  ( (TH1F*)((*hh).second) )->Write();
      if ( theTypes[(*hh).first] == 2 )  ( (TH2F*)((*hh).second) )->Write();
      if ( theTypes[(*hh).first] == 3 )  ( (TProfile*)((*hh).second) )->Write();
    }

  else 
    if ( hh != theHistos.end() ) { 
      if ( theTypes[name] == 1 )  ( (TH1F*)((*hh).second) )->Write();
      if ( theTypes[name] == 2 )  ( (TH2F*)((*hh).second) )->Write();
      if ( theTypes[name] == 3 )  ( (TProfile*)((*hh).second) )->Write();
    }

  else 
    std::cout << "Histos::put() : Histogram " 
	 << name << " does not exist. Nothing done" << std::endl;

  f->Write();
  f->Close();

}  

void
Histos::divide(const std::string& h1, const std::string& h2, const std::string& h3) {

  HistoItr hh1 = theHistos.find(h1);
  HistoItr hh2 = theHistos.find(h2);
  HistoItr hh3 = theHistos.find(h3);

  if ( hh1 == theHistos.end() ||
       hh2 == theHistos.end() ||
       hh3 != theHistos.end() ) {

    if ( hh1 == theHistos.end() ) 
      std::cout << "Histos::divide() : First histo " 
	   << h1 << " does not exist" << std::endl;

    if ( hh2 == theHistos.end() ) 
      std::cout << "Histos::divide() : Second histo " 
	   << h2 << " does not exist" << std::endl;

    if ( hh3 != theHistos.end() ) 
      std::cout << "Histos::divide() : Third histo " 
	   << h3 << " already exists" << std::endl;

  } else {

    if ( theTypes[h1] == 1 && theTypes[h2] == 1 ) { 
      
      theHistos[h3] = (TH1F*) ((*hh1).second)->Clone(h3.c_str());
      theTypes[h3] = 1;
      ((TH1F*)theHistos[h3])->Divide( (TH1F*)( (*hh2).second ) );

    }
      
    if ( theTypes[h1] == 2 && theTypes[h2] == 2 ) { 

      theHistos[h3] = (TH2F*)((*hh1).second)->Clone(h3.c_str());
      theTypes[h3] = 2;
      ((TH2F*)theHistos[h3])->Divide( (TH2F*)( (*hh2).second ) );
      
    }

  }

}

void Histos::addObject(const std::string& name, TObject * obj)
{
  HistoItr hh = theObjects.find(name);
  if (hh != theObjects.end())
    {
      std::cout << "FamosHistos::addObject() : Object " << name 
		<< " already exists" << std::endl;
      return;
    }
  // Potential source of memory leaks if not carefully used 
  theObjects.insert(std::pair<std::string,TObject*>(name,obj->Clone()));
}



void 
Histos::fill(const std::string& name, float val1, float val2,float val3) {

  //  std::cout << " Fill " << name << " " << val1 << " " << val2 << " " << val3 << std::endl;
  //  std::cout << &theHistos << std::endl;
  HistoItr hh = theHistos.find(name);
  //  std::cout << " Fill done " << std::endl;
  if ( hh == theHistos.end() ) {

    std::cout << "Histos::fill() : Histogram " << name 
	 << " does not exist" << std::endl;

  } else {

    if ( theTypes[name] == 1 ) 
      ( (TH1F*) ( (*hh).second ) )->Fill(val1,val2);

    if ( theTypes[name] == 2 ) 
      ( (TH2F*) ( (*hh).second ) )->Fill(val1,val2,val3);
    
    if ( theTypes[name] == 3 ) 
      ( (TProfile*) ( (*hh).second ) )->Fill(val1,val2,val3);
  }

}

void Histos::fillByNumber(const std::string& name,int number,float val1,float val2,float val3)
{
  std::ostringstream oss;
  oss << name << number;
  fill(oss.str(),val1,val2,val3);
}

void Histos::bookByNumber(const std::string& name, int n1,int n2,
			  int nx  , float xmin   , float xmax,
			  int ny  , float ymin,  float ymax)
{
  if(n1>n2)
    {
      std::cout <<" Histos: problem with bookByNumber - Do nothing" << std::endl;
    }
  for(int ih=n1;ih<=n2;++ih)
    {
       std::ostringstream oss;
       oss << name << ih;
       book(oss.str(),nx,xmin,xmax,ny,ymin,ymax);
    }

}
