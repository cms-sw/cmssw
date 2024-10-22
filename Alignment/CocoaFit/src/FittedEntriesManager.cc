//   COCOA class implementation file
//Id:  FittedEntriesManager.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "Alignment/CocoaFit/interface/FittedEntriesManager.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

FittedEntriesManager* FittedEntriesManager::instance = nullptr;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FittedEntriesManager* FittedEntriesManager::getInstance() {
  if (!instance) {
    instance = new FittedEntriesManager;
  }
  return instance;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ add a new set of fitted entries
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void FittedEntriesManager::AddFittedEntriesSet(FittedEntriesSet* fents) { theFittedEntriesSets.push_back(fents); }

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ dump data to histograms
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//tvoid FittedEntriesManager::WriteHeader()
void FittedEntriesManager::MakeHistos() {
  //----------- Get
  //----------- Loop entries
  //-  vfescite = theFittedEntriesSets.begin();
  //  std::vector< FittedEntry* > vfe = theFittedEntriesSets.begin()->FittedEntries();
  //  std::vector< FittedEntry* >::const_iterator vfecite2;
  //-- Number of fitted entries (equal for all Fitted Entries Sets )
  if (ALIUtils::debug >= 5)
    std::cout << "No sets2 " << theFittedEntriesSets.size() << " No ent "
              << ((*(theFittedEntriesSets.begin()))->FittedEntries()).size() << std::endl;
  std::ofstream fout;
  std::ofstream fout2;
  fout.open("fittedEntries.out");
  fout2.open("longFittedEntries.out");
  //---------- Dump dimensions
  ALIUtils::dumpDimensions(fout);
  ALIUtils::dumpDimensions(fout2);

  AddFittedEntriesSet(new FittedEntriesSet(theFittedEntriesSets));  //add a new set that averages all the others

  //---------- Loop sets of entries
  std::vector<FittedEntriesSet*>::const_iterator vfescite;
  std::vector<FittedEntry*>::const_iterator vfecite;
  ALIint jj = 1;
  for (vfescite = theFittedEntriesSets.begin(); vfescite != theFittedEntriesSets.end(); ++vfescite) {
    //---------- Loop entries
    if (vfescite == theFittedEntriesSets.begin()) {
      //----- dump entries names if first set
      fout << "  ";
      ALIint ii = 0;
      for (vfecite = ((*vfescite)->FittedEntries()).begin(); vfecite != ((*vfescite)->FittedEntries()).end();
           ++vfecite) {
        ALIstring filename = createFileName((*vfecite)->getOptOName(), (*vfecite)->getEntryName());
        fout << ii << ": " << std::setw(13) << filename << " ";
        ;
        if (ALIUtils::debug >= 3)
          std::cout << ii << ": " << std::setw(13) << filename << " = " << (*vfecite)->getName() << std::endl;
        if (ALIUtils::debug >= 3)
          std::cout << filename << " ";
        if (ALIUtils::debug >= 3)
          std::cout << "OPENING FITTED ENTRIES file " << filename << std::endl;
        ii++;
      }
      //      fout << std::setw(17) << "2:-4:";
      fout << std::endl;
      if (ALIUtils::debug >= 3)
        std::cout << std::endl;
    }
    //----- Dump entry set number
    fout << jj << " ";
    fout2 << jj << " ";
    //----- Dump measurements date
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if (gomgr->GlobalOptions()["DumpDateInFittedEntries"] >= 1) {
      fout << (*vfescite)->getDate() << " " << (*vfescite)->getTime() << " ";
      fout2 << (*vfescite)->getDate() << " " << (*vfescite)->getTime() << " ";
    }

    //ALIint ii = 0;
    for (vfecite = ((*vfescite)->FittedEntries()).begin(); vfecite != ((*vfescite)->FittedEntries()).end(); ++vfecite) {
      //      std::cout << ii << *vfescite << " FITTEDENTRY: "   << vfecite << " " <<*vfecite << " " << (*vfecite)->Value() << std::endl;
      //      if( ii == 2 || ii == 4 ) {
      fout << std::setprecision(8) << std::setw(10) << (*vfecite)->getValue() << " " << (*vfecite)->getSigma() << "  ";
      //- fout << std::setw(9) << std::setprecision(6) << (*vfecite)->getValue()  << " +- " << (*vfecite)->getSigma() << "  ";
      //      }
      if (ALIUtils::debug >= 3)
        std::cout << " FITTEDENTRY:" << std::setprecision(5) << std::setw(8) << (*vfecite)->getValue() << " +- "
                  << (*vfecite)->getSigma() << std::endl;

      ALIstring filename = createFileName((*vfecite)->getOptOName(), (*vfecite)->getEntryName());
      fout2 << std::setprecision(8) << std::setw(10) << filename << " " << (*vfecite)->getValue() << " "
            << (*vfecite)->getSigma() << "  ";
      //ii++;
    }
    //    dumpEntriesSubstraction( fout, *(*vfescite), 2, 4);
    fout << std::endl;
    fout2 << std::endl;
    if (ALIUtils::debug >= 3)
      std::cout << std::endl;
    jj++;
  }
  fout.close();
  fout2.close();

  GetDifferentBetweenLasers();
}

#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void FittedEntriesManager::GetDifferentBetweenLasers() {
  std::vector<OpticalObject*> optoList = Model::OptOList();
  std::vector<OpticalObject*>::const_iterator ite;
  std::map<ALIstring, LightRay*> lrays;

  for (ite = optoList.begin(); ite != optoList.end(); ++ite) {
    if ((*ite)->type() == "laser") {
      LightRay* lightray = new LightRay;
      lightray->startLightRay(*ite);
      lrays[(*ite)->parent()->name()] = lightray;
    }
  }

  std::map<ALIstring, LightRay*>::const_iterator lite1, lite2;
  for (lite1 = lrays.begin(); lite1 != lrays.end(); ++lite1) {
    lite2 = lite1;
    ++lite2;
    for (; lite2 != lrays.end(); ++lite2) {
      if (lite1 == lite2)
        continue;
      CLHEP::Hep3Vector dirdiff = ((*lite1).second->direction() - (*lite2).second->direction());
      if (ALIUtils::debug >= 0) {
        std::cout << "LASER DIFF " << (*lite1).first << " & " << (*lite2).first << " " << dirdiff.mag() * 180. / M_PI
                  << "o " << dirdiff.mag() << " rad " << dirdiff << std::endl;

        (*lite1).second->dumpData(ALIstring(" laser ") + (*lite1).first);
        (*lite2).second->dumpData(ALIstring(" laser ") + (*lite2).first);
      }
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ create file name to dump fitted entries values taking the last name of optoName and the full entryName with space converted to '.'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIstring FittedEntriesManager::createFileName(const ALIstring& optoName, const ALIstring& entryName) {
  //  std::cout << "in createFileName " << optoName << " " << entryName << std::endl;
  ALIstring filename;
  //-  std::cout << "o" << optoName << " e " << entryName << std::endl;
  /*  ALIint last_slash =  optoName.rfind('/');
  ALIint size = optoName.length();
  //-   std::cout << last_slash << " " << size << "optoname " << optoName << std::endl;

  filename = optoName.substr( last_slash+1, size );
  */
  //----- substitute '/' by '.' in opto name
  filename = optoName.substr(2, optoName.size());  // skip the first 's/'
  ALIint slash = -1;
  for (;;) {
    slash = filename.find('/', slash + 1);
    if (slash == -1)
      break;
    filename[slash] = '.';
  }

  //----- Check if there is a ' ' in entry (should not happen now)
  ALIint space = entryName.rfind(' ');
  filename.append(".");
  ALIstring en = entryName;
  if (space != -1)
    en[space] = '_';

  //----- Merge opto and entry names
  // now it is not used as filename   filename.append( en + ".out");
  filename.append(en);
  if (ALIUtils::debug >= 3)
    std::cout << "filename " << filename << std::endl;

  return filename;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ create file name to dump fitted entries values taking the last name of optoName and the full entryName with space converted to '-'
// entry1/2 are the entries name including OptO name
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void FittedEntriesManager::dumpEntriesSubstraction(std::ofstream& fout,
                                                   FittedEntriesSet& fes,
                                                   ALIint order1,
                                                   ALIint order2) {
  //---------- Found order of entry1 and entry2 in FittedEntriesSet fes
  // (the order will be the same for every FittedEntriesSet
  //std::vector< FittedEntriesSet* >::const_iterator vfescite = theFittedEntriesSets.begin();
  /* std::vector< FittedEntry* >::const_iterator vfecite;
  ALIint order1, order2;
  ALIint jj=0;
  for( vfecite = (fes.FittedEntries()).begin(); vfecite != (fes.FittedEntries()).end(); vfecite++) {
    ALIstring entryTemp = (*vfecite)->OptOName() + "/" + (*vfecite)->EntryName();
    if(entryTemp == entry1) order1 = jj;
    if(entryTemp == entry2) order2 = jj;
    jj++;
  }
  */

  FittedEntry* fe1 = *((fes.FittedEntries()).begin() + order1);
  FittedEntry* fe2 = *((fes.FittedEntries()).begin() + order2);
  //-------- Substract values of entry1 and entry2 (with errors)
  ALIdouble val1 = fe1->getValue();
  ALIdouble val2 = fe2->getValue();
  ALIdouble sig1 = fe1->getSigma();
  ALIdouble sig2 = fe2->getSigma();
  ALIdouble val = val1 - val2;
  ALIdouble sig = sqrt(sig1 * sig1 + sig2 * sig2);
  //-  std::cout << "CHECK " << val1 << " "<< val2 << " "<< sig1 << " "<< sig2 << std::endl;
  fout << std::setprecision(6) << std::setw(8) << val << " +- " << sig << "  ";
  if (ALIUtils::debug >= 3)
    std::cout << " FITTEDENTRY:" << std::setprecision(5) << std::setw(8) << val << " +- " << sig << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
/*void FittedEntriesManager::MakeHistos1()
{

  std::vector< FittedEntriesSet* >::const_iterator vfescite;
  std::vector< FittedEntry* >::const_iterator vfecite;
  //----------- Get 
  //----------- Loop entries
  vfescite = theFittedEntriesSets.begin(); 
  //  std::vector< FittedEntry* > vfe = theFittedEntriesSets.begin()->FittedEntries();
  //  std::vector< FittedEntry* >::const_iterator vfecite2;
  ALIint ii;
  //-- Number of fitted entries (equal for all Fitted Entries Sets 
  ALIint NoFitEnt = ( ( *(theFittedEntriesSets.begin()) )->FittedEntries() ).size();
  if( ALIUtils::debug >= 3) std::cout << "No sets1 " << theFittedEntriesSets.size() << " No ent " << NoFitEnt << std::endl;
  std::ofstream fout;
  ALIint jj;
  for( ii = 0; ii < NoFitEnt; ii++) {    
    jj = 1;

    for( vfescite = theFittedEntriesSets.begin(); vfescite != theFittedEntriesSets.end(); vfescite++) {
      vfecite = ((*vfescite)->FittedEntries()).begin() + ii;
      //----- create file name
      if( vfescite == theFittedEntriesSets.begin() ) {
	if( ALIUtils::debug >= 3) std::cout << " create filename " << (*vfecite)->OptOName() <<  (*vfecite)->EntryName() << std::endl;
        ALIstring filename = createFileName( (*vfecite)->OptOName(), (*vfecite)->EntryName() );  
	fout.open( filename.c_str() );
	if( ALIUtils::debug >= 3) std::cout << "OPENING FITTED ENTRIES file " << filename << std::endl;
      }
      //      std::cout << ii << *vfescite << " FITTEDENTRY: "   << vfecite << " " <<*vfecite << " " << (*vfecite)->Value() << std::endl;
      fout << jj << " " << (*vfecite)->Value()  << (*vfecite)->Sigma() << std::endl;   
      if( ALIUtils::debug >= 3) std::cout << ii << " FITTEDENTRY: " << (*vfecite)->OptOName() << " / " << (*vfecite)->EntryName() << " " << (*vfecite)->Value() << " " << (*vfecite)->Sigma() << std::endl;
      jj++;
    }
    fout.close();
  }

}
*/
