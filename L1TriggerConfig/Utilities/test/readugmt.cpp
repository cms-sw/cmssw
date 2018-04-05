#include "L1Trigger/L1TCommon/src/Setting.cc"
#include "L1Trigger/L1TCommon/src/Mask.cc"
#include "L1Trigger/L1TCommon/src/XmlConfigReader.cc"
#include "L1Trigger/L1TCommon/src/TrigSystem.cc"
#include "L1Trigger/L1TMuon/src/L1TMuonGlobalParamsHelper.cc"
#include "L1Trigger/L1TMuon/src/L1TMuonGlobalParams_PUBLIC.cc"

#include <iostream>
#include <fstream>

// To compile run these lines in your CMSSW_X_Y_Z/src/ :
/*
cmsenv
eval "export `scram tool info xerces-c | sed -n -e 's/INCLUDE=/XERC_INC=/gp'`"
eval "export `scram tool info xerces-c | sed -n -e 's/LIBDIR=/XERC_LIB=/gp'`"
eval "export `scram tool info boost    | sed -n -e 's/INCLUDE=/BOOST_INC=/gp'`"
eval "export `scram tool info boost    | sed -n -e 's/LIBDIR=/BOOST_LIB=/gp'`"
g++ -g -std=c++11 -o test readugmt.cpp -I./ -I$CMSSW_RELEASE_BASE/src -I$XERC_INC -L$XERC_LIB -lxerces-c -I$BOOST_INC -L$BOOST_LIB -lboost_thread -lboost_signals -lboost_date_time -L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/ -lFWCoreMessageLogger -lCondFormatsL1TObjects
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

    // finally, push all payloads to the XML parser and construct the trigSystem objects with each of those
    l1t::XmlConfigReader xmlRdr;
    l1t::TrigSystem trgSys;

// uncommenting the following line seems to solve one of the problems
//    trgSys.addProcRole("processors", "processors");

    // HW settings should always go first
    for(auto &name : sequence){
        cout<<"Parsing "<<name<<endl;
        xmlRdr.readDOMFromString( xmlPayload[name] );
        xmlRdr.readRootElement  ( trgSys           );
    }
    trgSys.setConfigured();

try {

//    map<string, l1t::Setting> conf = trgSys.getSettings("ugmt_processor");
//    map<string, l1t::Mask>    rs   = trgSys.getMasks   ("daqttcs");
//    string tmp = conf["bmtfInputsToDisable"].getValueAsStr();
//    cout << "bmtfInputsToDisable=" << tmp << endl;

    L1TMuonGlobalParamsHelper m_params_helper;
    m_params_helper.loadFromOnline(trgSys);

} catch ( std::runtime_error &e ){
cout << "Exception thrown: "<< e.what() << endl;
}

    return 0;
}

