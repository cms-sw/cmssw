//   COCOA class implementation file
//Id:  FittedEntriesSet.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include <fstream>
#include <map>
#include "Alignment/CocoaFit/interface/FittedEntriesSet.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Entry.h"

#ifdef MAT_MESCHACH
#include "Alignment/CocoaFit/interface/MatrixMeschach.h"
#endif
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
FittedEntriesSet::FittedEntriesSet( MatrixMeschach* AtWAMatrix )
{
  //- theTime = Model::MeasurementsTime();
  theDate = Measurement::getCurrentDate();
  theTime = Measurement::getCurrentTime();
  
  theMinEntryQuality = 2;
  theEntriesErrorMatrix = AtWAMatrix;
  
  Fill();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
FittedEntriesSet::FittedEntriesSet( std::vector<ALIstring> wl )
{
  //- theTime = Model::MeasurementsTime();
  theDate = wl[0];
  theTime = "99:99";
  
  theMinEntryQuality = 2;
  theEntriesErrorMatrix = (MatrixMeschach*)0;
  
  FillEntriesFromFile( wl );

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FittedEntriesSet::FittedEntriesSet( std::vector<FittedEntriesSet*> vSets )
{
  theDate = "99/99/99";
  theTime = "99:99";
  
  theMinEntryQuality = 2;
  theEntriesErrorMatrix = (MatrixMeschach*)0;
  
  FillEntriesAveragingSets( vSets );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::Fill( )
{

  FillEntries( );
  FillCorrelations( );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::FillEntries()
{
  //---------- Store Fitted Entries 
  //----- Iterate over entry list
  std::vector<Entry*>::const_iterator vecite; 
  for ( vecite = Model::EntryList().begin();
    vecite != Model::EntryList().end(); vecite++ ) {
    //--- Only for good quality parameters (='unk')
    if ( (*vecite)->quality() >= theMinEntryQuality ) {
      //      ALIdouble dimv =  (*vecite)->ValueDimensionFactor();
      //  ALIdouble dims =  (*vecite)->SigmaDimensionFactor();
      ALIint ipos = (*vecite)->fitPos();
      FittedEntry* fe = new FittedEntry( (*vecite), ipos, sqrt(theEntriesErrorMatrix->Mat()->me[ipos][ipos]));
      //-      std::cout << fe << "IN fit FE " << fe->theValue<< " " << fe->Sigma()<< " "  << sqrt(theEntriesErrorMatrix->Mat()->me[NoEnt][NoEnt]) / dims<< std::endl;
      theFittedEntries.push_back( fe );
    }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::FillCorrelations()
{
  //------ Count the number of entries that will be in the set
  ALIuint nent = 0;
  std::vector<Entry*>::const_iterator vecite;
  for ( vecite = Model::EntryList().begin();
	         vecite != Model::EntryList().end(); vecite++ ) {
    if((*vecite)->quality() > theMinEntryQuality ) {
      nent++;
    }
  }
      
  CreateCorrelationMatrix( nent );
  //---------- Store correlations
  ALIuint ii;
  for( ii = 0; ii < nent; ii++) {
    for(ALIuint jj = ii+1; jj < nent; jj++) {
      ALIdouble corr = theEntriesErrorMatrix->Mat()->me[ii][jj];
      if (corr != 0) {
	corr /= ( sqrt(theEntriesErrorMatrix->Mat()->me[ii][ii])
	  / sqrt(theEntriesErrorMatrix->Mat()->me[jj][jj]) );
	theCorrelationMatrix[ii][jj] = corr;
      }
    }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::CreateCorrelationMatrix( const ALIuint nent )
{

  std::vector<ALIdouble> vd( nent, 0.);
  std::vector< std::vector<ALIdouble> > vvd( nent, vd);
  theCorrelationMatrix = vvd;

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::FillEntriesFromFile( std::vector<ALIstring> wl)
{

  ALIuint siz = wl.size();
  for( ALIuint ii = 1; ii< siz; ii+=3 ) {
    FittedEntry* fe = new FittedEntry( wl[ii], ALIUtils::getFloat(wl[ii+1]), ALIUtils::getFloat(wl[ii+2]));
    theFittedEntries.push_back( fe );
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::FillEntriesAveragingSets( std::vector<FittedEntriesSet*> vSets )
{

  std::vector<FittedEntry*> vFEntry;
  ALIuint nEntry = vSets[0]->FittedEntries().size();
  //  ALIuint setssiz = vSets.size();
  for( ALIuint ii = 0; ii < nEntry; ii++ ){  // loop to FittedEntry's
    if(ALIUtils::debug >= 5) std::cout << "FillEntriesAveragingSets entry " << ii << std::endl;
    vFEntry.clear();
    for( ALIuint jj = 0; jj < vSets.size(); jj++ ){  // look for FittedEntry ii in each Sets
     if(ALIUtils::debug >= 5) std::cout << "FillEntriesAveragingSets set " << jj << std::endl;
      //----- Check all have the same number of entries
      if( vSets[jj]->FittedEntries().size() != nEntry ){
	std::cerr << "!!! FATAL ERROR FittedEntriesSet::FillEntriesAveragingSets  set number " << jj 
		  << " has different number of entries = " 
		  << vSets[jj]->FittedEntries().size() 
		  << " than first set = " << nEntry << std::endl;
	exit(1);
      }
      
      vFEntry.push_back( vSets[jj]->FittedEntries()[ii] );
    }
   FittedEntry* fe = new FittedEntry( vFEntry );
   if(ALIUtils::debug >= 5) std::cout << "FillEntriesAveragingSets new fentry " << fe->getValue() << " " << fe->getSigma() << std::endl;
   theFittedEntries.push_back( fe );
  }


}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntriesSet::SetOptOEntries()
{
  if(ALIUtils::debug >= 5) std::cout << "  FittedEntriesSet::SetOptOEntries " << theFittedEntries.size() << std::endl;

  std::vector< FittedEntry* >::const_iterator ite;
  for( ite = theFittedEntries.begin();ite != theFittedEntries.end();ite++){
    FittedEntry* fe = (*ite);
    OpticalObject * opto = Model::getOptOByName( fe->getOptOName() );
    Entry* entry = Model::getEntryByName( fe->getOptOName(), fe->getEntryName() );
    entry->setValue( fe->getValue() );
    entry->setSigma( fe->getSigma() );

   if(ALIUtils::debug >= 5) std::cout << "  FittedEntriesSet::SetOptOEntries() " << opto->name() << " " << entry->name() << std::endl;
    opto->setGlobalCoordinates();
    opto->setOriginalEntryValues();
 }
}
