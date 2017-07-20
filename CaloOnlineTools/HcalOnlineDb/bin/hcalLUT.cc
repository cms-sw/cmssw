#include <iostream>
#include <stdexcept>
#include <sstream>
#include "TString.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"

using namespace std;

void mergeLUTs(const char* flist, const char* out){
    LutXml xmls;
    stringstream ss(flist);
    while (ss.good()){
	string file;
        ss >> file;
	xmls += LutXml(file);
    }
    xmls.write(out);
}

void diffLUTs(const char* file1, const char* file2, int verbosity=0){
    edm::FileInPath xmlFile1(file1);
    edm::FileInPath xmlFile2(file2);


    LutXml xmls1(xmlFile1.fullPath());
    LutXml xmls2(xmlFile2.fullPath());

    xmls1.create_lut_map();
    xmls2.create_lut_map();

    const char* DET[7]={"EMPTY", "HB" , "HE", "HO", "HF", "HT", "OTHER"};
    int ntota[7]; 
    int nzero[7]; 
    int nreal[7]; 
    int extra[7];
    for(int i=0; i<7; ++i) {
	ntota[i]=0;
	nzero[i]=0;
	nreal[i]=0;
	extra[i]=0;
    }
    cout << file2 << endl;
    for (LutXml::const_iterator xml2 = xmls2.begin(); xml2 != xmls2.end(); ++xml2){
	HcalGenericDetId id(xml2->first);
	LutXml::const_iterator xml1 =xmls1.find(id.rawId());
	HcalGenericDetId::HcalGenericSubdetector subdet = id.genericSubdet();
	ntota[subdet]++;
	if(xml1==xmls1.end()){
	    extra[subdet]++;
	    cout << "FAIL: DetId " << id << "(" << id.rawId() <<") IS PRESENT IN " << file2 << "  BUT ABSENT IN " <<  file1  << " " << id << endl;
	    //return;
	}
	bool zero=true;
	const std::vector<unsigned int>& lut2 = xml2->second;
	for(size_t i=0; i<lut2.size(); ++i){
	    if(lut2[i]>0){
		zero=false;
	    }
	}
	if(zero) nzero[subdet]++;
	else     nreal[subdet]++;
    }
    cout << Form("%3s:  %8s  %8s  %8s  %8s", "Det", "total", "nonzeros", "zeroes", "extra") << endl; 
    for(int i=1; i<6; ++i) cout << Form("%3s:  %8d  %8d  %8d  %8d", DET[i], ntota[i], nreal[i], nzero[i], extra[i]) << endl;
    cout << "--------------------------------------------" << endl;
    for(int i=0; i<7; ++i) {
	ntota[i]=0;
	nzero[i]=0;
	nreal[i]=0;
	extra[i]=0;
    }
    cout << file1 << endl;
    for (LutXml::const_iterator xml1 = xmls1.begin(); xml1 != xmls1.end(); ++xml1){
	HcalGenericDetId id(xml1->first);
	LutXml::const_iterator xml2 =xmls2.find(id.rawId());
	HcalGenericDetId::HcalGenericSubdetector subdet = id.genericSubdet();
	ntota[subdet]++;
	if(xml2==xmls2.end()){
	    extra[subdet]++;
	    HcalDetId hid(id);
	    cout << "FAIL: DetId " << hid << "(" << id.rawId() <<") IS PRESENT IN " << file1 << "  BUT ABSENT IN " <<  file2  << " " << id << endl;
	}
	bool zero=true;
	const std::vector<unsigned int>& lut1 = xml1->second;
	for(size_t i=0; i<lut1.size(); ++i){
	    if(lut1[i]>0){
		zero=false;
		break;
	    }
	}
	if(zero) nzero[subdet]++;
	else     nreal[subdet]++;
    }
    cout << Form("%3s:  %8s  %8s  %8s  %8s", "Det", "total", "nonzeros", "zeroes", "extra") << endl; 
    for(int i=1; i<6; ++i) cout << Form("%3s:  %8d  %8d  %8d  %8d", DET[i], ntota[i], nreal[i], nzero[i], extra[i]) << endl;
    cout << "--------------------------------------------" << endl;

    for(int i=0; i<7; ++i) {
	ntota[i]=0;
	nzero[i]=0;
	nreal[i]=0;
    }
    for (LutXml::const_iterator xml1 = xmls1.begin(); xml1 != xmls1.end(); ++xml1){
	HcalGenericDetId id(xml1->first);
	LutXml::const_iterator xml2 =xmls2.find(id.rawId());
	if(xml2==xmls2.end()){
	    continue;
	}

	const std::vector<unsigned int>& lut1 = xml1->second;
	const std::vector<unsigned int>& lut2 = xml2->second;

	size_t size = lut1.size();
	bool match=true;
	if(size != lut2.size()) {
	    match=false;
	}

	for(size_t i=0; i<size && match; ++i){
	    if(lut1[i]!=lut2[i]) {
		match=false;
		if(verbosity>0){
		    cout << Form("Mismatach in index=%3d, %4d!=%4d, ", int(i), lut1[i], lut2[i]) << id << endl;
		}
	    }
	}
	HcalGenericDetId::HcalGenericSubdetector subdet = id.genericSubdet();
	ntota[subdet]++;
	if(match) nreal[subdet]++;
	else      nzero[subdet]++;
    }
    string result="PASS!";
    for(int i=0; i<7; ++i) if(nzero[i]>0) result="FAIL!";
    cout << "Comparison:" <<endl;
    cout << Form("%3s:  %8s  %8s  %8s", "Det", "total", "match", "mismatch") << endl; 
    for(int i=1; i<6; ++i) cout << Form("%3s:  %8d  %8d  %8d", DET[i], ntota[i], nreal[i], nzero[i]) << endl;
    cout << "--------------------------------------------" << endl;
    cout << result << endl;
}


int main(int argc, char ** argv){

    optutl::CommandLineParser parser("runTestParameters");
    parser.parseArguments (argc, argv, true);
    if(argc<2){
	std::cerr << "runTest: missing input command" << std::endl;
    }
    else if(strcmp(argv[1],"merge")==0){
	std::string flist_	    = parser.stringValue("storePrepend");
	std::string out_	    = parser.stringValue("outputFile");
	mergeLUTs(flist_.c_str(),  out_.c_str());
    }
    else if (strcmp(argv[1],"diff")==0){
	std::vector<std::string> inputFiles_ = parser.stringVector("inputFiles");
	diffLUTs(inputFiles_[0].c_str(), inputFiles_[1].c_str(), parser.integerValue("section")); 
    }
    else if (strcmp(argv[1],"create-lut-loader")==0){
	std::string _file_list = parser.stringValue("outputFile");
	std::string _tag     = parser.stringValue("tag");
	std::string _comment = parser.stringValue("storePrepend");
	std::string _prefix  = "HCAL";
	std::string _version = "Physics";
	int _subversion	     = 0; 
	HcalLutManager manager;
	manager.create_lut_loader( _file_list, _prefix, _tag, _comment, _tag, _subversion);
    }
    else {
	throw std::invalid_argument( Form("Unknown command: %s", argv[1]) );
    }

    return 0;
}

