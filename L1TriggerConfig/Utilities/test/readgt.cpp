#include "L1Trigger/L1TCommon/src/Setting.cc"
#include "L1Trigger/L1TCommon/src/Mask.cc"
#include "L1Trigger/L1TCommon/src/Tools.cc"
#include "L1Trigger/L1TCommon/src/XmlConfigReader.cc"
//#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"
#include <iostream>
#include <fstream>
#include <algorithm>
// To compile run these lines in your CMSSW_X_Y_Z/src/ :
/*
cmsenv
eval "export `scram tool info xerces-c | sed -n -e 's/INCLUDE=/XERC_INC=/gp'`"
eval "export `scram tool info xerces-c | sed -n -e 's/LIBDIR=/XERC_LIB=/gp'`"
eval "export `scram tool info boost    | sed -n -e 's/INCLUDE=/BOOST_INC=/gp'`"
eval "export `scram tool info boost    | sed -n -e 's/LIBDIR=/BOOST_LIB=/gp'`"
g++ -g -std=c++11 -o test readgt.cpp -I./ -I$CMSSW_RELEASE_BASE/src -I$CMSSW_BASE/src -I$XERC_INC -L$XERC_LIB -lxerces-c -I$BOOST_INC -L$BOOST_LIB -lboost_thread -lboost_signals -lboost_date_time -L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/ -lFWCoreMessageLogger -L$CMSSW_BASE/lib/$SCRAM_ARCH/ -lL1TriggerL1TCommon -lCondFormatsL1TObjects
*/

using namespace std;

int main(int argc, char *argv[]){
    // read the input xml file into a string
    list<string> sequence;
    map<string,string> xmlPayload;
    for(int p=1; p<argc; p++){

        ifstream input( argv[p] );
        if( !input ){ cout << "Cannot open " << argv[p] << " file" << endl; return 0; }
        sequence.push_back( argv[p] );

        size_t nLinesRead=0;

        while( !input.eof() ){
            string tmp;
            getline( input, tmp, '\n' );
            xmlPayload[ argv[p] ].append( tmp );
            nLinesRead++;
        }

        cout << argv[p] << ": read " << nLinesRead << " lines" << endl;
        input.close();
    }

  list<string>::const_iterator fname = sequence.begin();
  std::cout << "Prescales: " << *fname << std::endl;
  std::string xmlPayload_prescale    = xmlPayload[*fname++];
  std::cout << "Finor: " << *fname << std::endl;
  std::string xmlPayload_mask_finor  = xmlPayload[*fname++];
  std::cout << "Veto: " << *fname << std::endl;
  std::string xmlPayload_mask_veto   = xmlPayload[*fname++];
  std::cout << "AlgoBX: " << *fname << std::endl;
  std::string xmlPayload_mask_algobx = xmlPayload[*fname++];

  unsigned int m_numberPhysTriggers = 512;
  unsigned int m_bx_mask_default = 1;


  std::vector<std::vector<int> > prescales;
  std::vector<unsigned int> triggerMasks;
  std::vector<int> triggerVetoMasks;
  std::map<int, std::vector<int> > triggerAlgoBxMaskAlgoTrig;


  // Prescales
    l1t::XmlConfigReader xmlReader_prescale;
    l1t::TrigSystem ts_prescale;
    ts_prescale.addProcRole("uGtProcessor", "uGtProcessor");

std::cout<<"!!!!!!!!!!!!2"<<std::endl;
    // run the parser 
    xmlReader_prescale.readDOMFromString( xmlPayload_prescale ); // initialize it
    xmlReader_prescale.readRootElement( ts_prescale, "uGT" ); // extract all of the relevant context
    ts_prescale.setConfigured();

std::cout<<"!!!!!!!!!!!!3"<<std::endl;
    std::map<std::string, l1t::Setting> settings_prescale = ts_prescale.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_prescale = settings_prescale["prescales"].getTableRows();

    for(std::map<std::string, l1t::Setting>::const_iterator i=settings_prescale.begin(); i!=settings_prescale.end(); i++)
        std::cout<<i->first<<std::endl;

std::cout<<"!!!!!!!!!!!!4"<<std::endl;
    unsigned int numColumns_prescale = 0;
    if( tRow_prescale.size()>0 ){
      std::vector<std::string> firstRow_prescale = tRow_prescale[0].getRow();
      numColumns_prescale = firstRow_prescale.size();
    } else {
      std::cout<<" tRow_prescale.size = 0"<<std::endl;
    }
    
std::cout<<"!!!!!!!!!!!!5"<<std::endl;
    int NumPrescaleSets = numColumns_prescale - 1;
    int NumAlgos_prescale = 0;//*( std::max_element(std::begin(tRow_prescale), std::end(tRow_prescale)) );
    for( auto it=tRow_prescale.begin(); it!=tRow_prescale.end(); it++ ){
        unsigned int algoBit = it->getRowValue<unsigned int>("algo/prescale-index");
        if( NumAlgos_prescale < algoBit+1 ) NumAlgos_prescale = algoBit+1;
    }

std::cout << "numColumns_prescale = " << numColumns_prescale << " NumAlgos_prescale = " << NumAlgos_prescale << std::endl;

    if( NumPrescaleSets > 0 ){
      // Fill default prescale set
      for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	prescales.push_back(std::vector<int>());
	for( int iBit = 0; iBit < NumAlgos_prescale; ++iBit ){
	  int inputDefaultPrescale = 1;
	  prescales[iSet].push_back(inputDefaultPrescale);
	}
      }

std::cout<<"!!!!!!!!!!!!6"<<std::endl;
      for( auto it=tRow_prescale.begin(); it!=tRow_prescale.end(); it++ ){
	unsigned int algoBit = it->getRowValue<unsigned int>("algo/prescale-index");
std::cout<<"algoBit = "<<algoBit<<std::endl;
	for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	  int prescale = it->getRowValue<unsigned int>(std::to_string(iSet));
	  prescales[iSet][algoBit] = prescale;
	}
      }
    }


std::cout<<"!!!!!!!!!!!!7"<<std::endl;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finor mask
  // Setting mask to default 1 (unmask)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerMasks.push_back(1);

std::cout<<"!!!!!!!!!!!!8"<<std::endl;
  
    l1t::XmlConfigReader xmlReader_mask_finor;
    l1t::TrigSystem ts_mask_finor;
    ts_mask_finor.addProcRole("uGtProcessor", "uGtProcessor");

std::cout<<"!!!!!!!!!!!!9"<<std::endl;
    // run the parser 
    xmlReader_mask_finor.readDOMFromString( xmlPayload_mask_finor ); // initialize it
    xmlReader_mask_finor.readRootElement( ts_mask_finor, "uGT" ); // extract all of the relevant context
    ts_mask_finor.setConfigured();

std::cout<<"!!!!!!!!!!!!10"<<std::endl;
    std::map<std::string, l1t::Setting> settings_mask_finor = ts_mask_finor.getSettings("uGtProcessor");

std::cout<<"!!!!!!!!!!!!11"<<std::endl;
    std::vector<l1t::TableRow> tRow_mask_finor = settings_mask_finor["finorMask"].getTableRows();

std::cout<<"!!!!!!!!!!!!12"<<std::endl;
    for( auto it=tRow_mask_finor.begin(); it!=tRow_mask_finor.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int mask = it->getRowValue<unsigned int>("mask");
      if( algoBit < m_numberPhysTriggers ) triggerMasks[algoBit] = mask;
    }

std::cout<<"!!!!!!!!!!!!13"<<std::endl;
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // veto mask
  // Setting veto mask to default 0 (no veto)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerVetoMasks.push_back(0);
  
std::cout<<"!!!!!!!!!!!!14"<<std::endl;
    l1t::XmlConfigReader xmlReader_mask_veto;
    l1t::TrigSystem ts_mask_veto;
    ts_mask_veto.addProcRole("uGtProcessor", "uGtProcessor");

std::cout<<"!!!!!!!!!!!!15"<<std::endl;
    // run the parser 
    xmlReader_mask_veto.readDOMFromString( xmlPayload_mask_veto ); // initialize it
    xmlReader_mask_veto.readRootElement( ts_mask_veto, "uGT" ); // extract all of the relevant context
    ts_mask_veto.setConfigured();

std::cout<<"!!!!!!!!!!!!16"<<std::endl;
    std::map<std::string, l1t::Setting> settings_mask_veto = ts_mask_veto.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_veto = settings_mask_veto["vetoMask"].getTableRows();

    for( auto it=tRow_mask_veto.begin(); it!=tRow_mask_veto.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int veto = it->getRowValue<unsigned int>("veto");
      if( algoBit < m_numberPhysTriggers ) triggerVetoMasks[algoBit] = int(veto);
    }

std::cout<<"!!!!!!!!!!!!17"<<std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Algo bx mask
    l1t::XmlConfigReader xmlReader_mask_algobx;
    l1t::TrigSystem ts_mask_algobx;
    ts_mask_algobx.addProcRole("uGtProcessor", "uGtProcessor");

std::cout<<"!!!!!!!!!!!!18"<<std::endl;
    // run the parser 
    xmlReader_mask_algobx.readDOMFromString( xmlPayload_mask_algobx ); // initialize it
    xmlReader_mask_algobx.readRootElement( ts_mask_algobx, "uGT" ); // extract all of the relevant context
    ts_mask_algobx.setConfigured();

std::cout<<"!!!!!!!!!!!!19"<<std::endl;
    std::map<std::string, l1t::Setting> settings_mask_algobx = ts_mask_algobx.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_algobx = settings_mask_algobx["algorithmBxMask"].getTableRows();

std::cout<<"!!!!!!!!!!!!20"<<std::endl;
    unsigned int numCol_mask_algobx = 0;
    if( tRow_mask_algobx.size()>0 ){
      std::vector<std::string> firstRow_mask_algobx = tRow_mask_algobx[0].getRow();
      numCol_mask_algobx = firstRow_mask_algobx.size();
    }
    
std::cout<<"!!!!!!!!!!!!21"<<std::endl;
    int NumAlgoBitsInMask = numCol_mask_algobx - 1;
    if( NumAlgoBitsInMask > 0 ){
      for( auto it=tRow_mask_algobx.begin(); it!=tRow_mask_algobx.end(); it++ ){
	int bx = it->getRowValue<unsigned int>("bx/algo");
	std::vector<int> maskedBits;
	for( int iBit=0; iBit<NumAlgoBitsInMask; iBit++ ){
	  unsigned int maskBit = it->getRowValue<unsigned int>(std::to_string(iBit));
	  if( maskBit!=m_bx_mask_default ) maskedBits.push_back(iBit);
	}
	if( maskedBits.size()>0 ) triggerAlgoBxMaskAlgoTrig[bx] = maskedBits;
      }
    }


std::cout<<"!!!!!!!!!!!!22"<<std::endl;
  // Set prescales to zero if masked
  for( unsigned int iSet=0; iSet < prescales.size(); iSet++ ){
    for( unsigned int iBit=0; iBit < prescales[iSet].size(); iBit++ ){
      // Add protection in case prescale table larger than trigger mask size
      if( iBit >= triggerMasks.size() ){
            edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
	      << "\nWarning: algoBit in prescale table >= triggerMasks.size() "
	      << "\nWarning: no information on masking bit or not, setting as unmasked "
	      << std::endl;
      }
      else {
	prescales[iSet][iBit] *= triggerMasks[iBit];
      }
    }
  }



    return 0;
}

