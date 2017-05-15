#include "L1Trigger/L1TCommon/src/Parameter.cc"
#include "L1Trigger/L1TCommon/src/Mask.cc"
#include "L1Trigger/L1TCommon/src/XmlConfigParser.cc"
#include "L1Trigger/L1TCommon/src/TriggerSystem.cc"

#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <map>

// To compile run these lines in your CMSSW_X_Y_Z/src/ :
/*
cmsenv
eval "setenv `scram tool info xerces-c | sed -n -e 's/INCLUDE=/XERC_INC /gp'`"
eval "setenv `scram tool info xerces-c | sed -n -e 's/LIBDIR=/XERC_LIB /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/INCLUDE=/BOOST_INC /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/LIBDIR=/BOOST_LIB /gp'`"
g++ -g -std=c++11 -o test readcalol1.cpp -I./ -I$CMSSW_RELEASE_BASE/src -I$XERC_INC -L$XERC_LIB -lxerces-c -I$BOOST_INC -L$BOOST_LIB -lboost_thread -lboost_signals -lboost_date_time -L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/ -lFWCoreMessageLogger -lCondFormatsL1TObjects
*/

using namespace std;

int main(int argc, char *argv[]){
    if( argc < 3 ) return 0;
    // read the input xml files into a map of string

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
    XmlConfigParser xmlReader;
    l1t::TriggerSystem ts;

    cout << "Parsing " << sequence.front() << endl;
if( false ){
    xmlReader.readDOMFromString( xmlPayload[sequence.front()] );
    xmlReader.readRootElement  ( ts, "calol1" );
} else {
    ts.addProcessor("processor0", "processors","-1","-1");
    ts.addProcessor("processor1", "processors","-1","-2");
    ts.addProcessor("processor2", "processors","-1","-3");
    ts.addProcessor("processor3", "processors","-1","-4");
    ts.addProcessor("processor4", "processors","-1","-5");
    ts.addProcessor("processor5", "processors","-1","-6");
    ts.addProcessor("processor6", "processors","-1","-7");
    ts.addProcessor("processor7", "processors","-1","-8");
    ts.addProcessor("processor8", "processors","-1","-9");
    ts.addProcessor("processor9", "processors","-1","-10");
    ts.addProcessor("processor10", "processors","-1","-11");
    ts.addProcessor("processor11", "processors","-1","-12");
    ts.addProcessor("processor12", "processors","-1","-13");
    ts.addProcessor("processor13", "processors","-1","-14");
    ts.addProcessor("processor14", "processors","-1","-15");
    ts.addProcessor("processor15", "processors","-1","-16");
    ts.addProcessor("processor16", "processors","-1","-17");
    ts.addProcessor("processor17", "processors","-1","-18");

    ts.addProcessor("CTP7_Phi0", "processors","-2","-0");
    ts.addProcessor("CTP7_Phi1", "processors","-2","-1");
    ts.addProcessor("CTP7_Phi2", "processors","-2","-2");
    ts.addProcessor("CTP7_Phi3", "processors","-2","-3");
    ts.addProcessor("CTP7_Phi4", "processors","-2","-4");
    ts.addProcessor("CTP7_Phi5", "processors","-2","-5");
    ts.addProcessor("CTP7_Phi6", "processors","-2","-6");
    ts.addProcessor("CTP7_Phi7", "processors","-2","-7");
    ts.addProcessor("CTP7_Phi8", "processors","-2","-8");
    ts.addProcessor("CTP7_Phi9", "processors","-2","-9");
    ts.addProcessor("CTP7_Phi10","processors","-2","-10");
    ts.addProcessor("CTP7_Phi11","processors","-2","-11");
    ts.addProcessor("CTP7_Phi12","processors","-2","-12");
    ts.addProcessor("CTP7_Phi13","processors","-2","-13");
    ts.addProcessor("CTP7_Phi14","processors","-2","-14");
    ts.addProcessor("CTP7_Phi15","processors","-2","-15");
    ts.addProcessor("CTP7_Phi16","processors","-2","-16");
}

    cout << "Parsing " << sequence.back() << endl;
    xmlReader.readDOMFromString( xmlPayload[sequence.back()] );
    xmlReader.readRootElement  ( ts, "calol1" );

    ts.setConfigured();

//    for(auto &q : ts.getProcToRoleAssignment()) cout << q.first << " - " << q.second << endl;

    // feel free to play with the containers:
    map<string, l1t::Parameter> conf = ts.getParameters("CTP7_Phi15"); // use your context id here - Layer1Processor
//    map<string, l1t::Mask>    rs   = ts.getMasks   ("processors"); // don't call a context that doesn't exist

    string layer1ECalScaleFactors= conf["layer1ECalScaleFactors"].getValueAsStr();
    string layer1HCalScaleFactors= conf["layer1HCalScaleFactors"].getValueAsStr();
    string layer1HFScaleFactors  = conf["layer1HFScaleFactors"].getValueAsStr();
    string layer1ECalScaleETBins = conf["layer1ECalScaleETBins"].getValueAsStr();
    string layer1HCalScaleETBins = conf["layer1HCalScaleETBins"].getValueAsStr();
    string layer1HFScaleETBins   = conf["layer1HFScaleETBins"].getValueAsStr();

    cout << "layer1ECalScaleFactors=" << layer1ECalScaleFactors << endl;
    cout << "layer1HCalScaleFactors=" << layer1HCalScaleFactors << endl;
    cout << "layer1HFScaleFactors  =" << layer1HFScaleFactors  << endl;
    cout << "layer1ECalScaleETBins =" << layer1ECalScaleETBins << endl;
    cout << "layer1HCalScaleETBins =" << layer1HCalScaleETBins << endl;
    cout << "layer1HFScaleETBins   =" << layer1HFScaleETBins   << endl;

    return 0;
}

