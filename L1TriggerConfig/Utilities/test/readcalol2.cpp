#include "L1Trigger/L1TCommon/src/Setting.cc"
#include "L1Trigger/L1TCommon/src/Mask.cc"
#include "L1Trigger/L1TCommon/src/XmlConfigReader.cc"
#include "L1Trigger/L1TCommon/src/TrigSystem.cc"

#include <iostream>
#include <fstream>

// To compile run these lines in your CMSSW_X_Y_Z/src/ :
/*
cmsenv
eval "setenv `scram tool info xerces-c | sed -n -e 's/INCLUDE=/XERC_INC /gp'`"
eval "setenv `scram tool info xerces-c | sed -n -e 's/LIBDIR=/XERC_LIB /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/INCLUDE=/BOOST_INC /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/LIBDIR=/BOOST_LIB /gp'`"
g++ -g -std=c++11 -o test readcalol2.cpp -I./ -I$CMSSW_RELEASE_BASE/src -I$XERC_INC -L$XERC_LIB -lxerces-c -I$BOOST_INC -L$BOOST_LIB -lboost_thread -lboost_signals -lboost_date_time -L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/ -lFWCoreMessageLogger -lCondFormatsL1TObjects -L$CMSSW_BASE/lib/slc6_amd64_gcc530/ -lL1TriggerL1TCommon
./test ~kkotov/public/MPs*.xml
*/

using namespace std;

int main(int argc, char *argv[]){
    if( argc < 2 ) return 0;

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

    // parse the string using the XML reader
    XmlConfigReader xmlRdr;
    l1t::TrigSystem trgSys;
    trgSys.addProcRole("MainProcessor", "processors"); //

    for(auto &name : sequence){
        cout<<"Parsing "<<name<<endl;
        xmlRdr.readDOMFromString( xmlPayload[name] );
        xmlRdr.readRootElement  ( trgSys, "calol2" );
    }
    trgSys.setConfigured();

    // feel free to play with the containers:
    map<string, l1t::Setting> conf = trgSys.getSettings("MainProcessor"); // use your context id here
//    map<string, l1t::Mask>    rs   = trgSys.getMasks   ("processors"); // don't call a context that doesn't exist

    string tmp = conf["leptonSeedThreshold"].getValueAsStr();
    cout << "leptonSeedThreshold=" << tmp << endl;

    return 0;
}

