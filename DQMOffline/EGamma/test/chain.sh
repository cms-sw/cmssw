destdir=/afs/cern.ch/cms/Physics/egamma/www/validation/312PreProd/Photons/DQMOffline/
echo Destination for plots is $destdir
fixedname=`echo $1 | sed 's/\/$//'`
inputdir=`echo $fixedname`
echo Input directory is $inputdir
shortname=`echo $fixedname | sed 's/.*\///'`
echo Nickname is $shortname
rootfile=`echo $shortname.root`
echo ROOT file containing plots will be $rootfile

rfdir $inputdir | grep 'root' | awk '{print $9}' | sed "s,^,$inputdir/," | xargs hadd -f $rootfile

#rm ../python/photonOfflineClient_cfi.py
rm photonOfflineClient_cfi.template
#    InputFileName = cms.untracked.string("file:__FILENAME__"),
sed "s%InputFileName = cms\.untracked\.string(\".*\")%InputFileName = cms.untracked.string(\"$rootfile\")%" ../python/photonOfflineClient_cfi.py > photonOfflineClient_cfi.template
mv -f photonOfflineClient_cfi.template ../python/photonOfflineClient_cfi.py
cmsRun PhotonOfflineClient_cfg.py

rm plotlist.txt
cat >> plotlist.txt << EOF
/DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV
/DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV/Conversions
/DQMData/Egamma/PhotonAnalyzer/BackgroundPhotons/Et above 0 GeV
/DQMData/Egamma/PhotonAnalyzer/BackgroundPhotons/Et above 0 GeV/Conversions
/DQMData/Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV
/DQMData/Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/Conversions
/DQMData/Egamma/PhotonAnalyzer/InvMass
/DQMData/Egamma/PhotonAnalyzer/Efficiencies
EOF

rm plotScript.C
cat >> plotScript.C << EOF
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TClass.h"
#include "TList.h"
#include "THStack.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TPaveStats.h"
#include "TSystem.h"

#include <fstream>
#include <iostream>


void plotScript(string file1) {

    ifstream infile("plotlist.txt");

    vector<string> plotDirectories;

    string s;
    
    while (getline(infile,s)) {
        plotDirectories.push_back(s);
    }

    
    ofstream outfile;

    string ofname = file1;

    while (ofname.find("/") != string::npos) {
        ofname.replace(ofname.find("/"), 1, "_");
    }
    while (ofname.find(".") != string::npos) {
        ofname.replace(ofname.find("."), 1, "_");
    }
    
    cout << ofname << endl;
    string prefix = ofname;
    ofname += ".html";

    outfile.open(ofname.c_str());
    outfile << "<!DOCTYPE html> <html> <head> <title>\n";
    outfile << file1 << "\n";
    outfile << "</title> </head>\n";
    outfile << "<body>\n";
    outfile << "<h1>\n";
    outfile << "<span style=\"color:#0404B4\">" << file1 << "</span>";
    outfile << "</h1>\n";
    outfile << "<a name=\"top\">Directory listing:</a><br>\n";
    int anchor = 0;
    for (vector<string>::iterator vecIt = plotDirectories.begin();
            vecIt != plotDirectories.end(); ++vecIt) {
        outfile << "<a href=\"#" << anchor << "\">" << *vecIt << "</a><br>\n";
        ++anchor;
    }
    
    TFile * f1 = new TFile(file1.c_str(), "READ");
    
    TDirectory * mydir; 
    TObject * obj;
    TCanvas * c0 = new TCanvas("c0","c0");
    anchor = 0;
    TLegend * leg = new TLegend(0.78,0.9,0.98,0.98);
    TH1F * firstH = new TH1F();
    firstH->SetLineWidth(2);
    firstH->SetLineColor(4);
    leg->AddEntry(firstH,file1.c_str(),"l");
    for (vector<string>::iterator vecIt = plotDirectories.begin();
            vecIt != plotDirectories.end(); ++vecIt) {
        outfile << "<br><a name=\"" << anchor << "\"><h2>" << *vecIt << "</h2></a> <a href=\"#top\">[Back to top]</a><br>\n";
        ++anchor;
        TH1 *h;
        TKey *key;
        mydir = f1->GetDirectory((*vecIt).c_str());
        TIter nextkey(mydir->GetListOfKeys());
        float scale = 1.;
        while (key = (TKey*)nextkey()) {
            obj = key->ReadObj();
            if (obj->IsA()->InheritsFrom("TH1")) {
                h = (TH1*)obj; 

                if (string(h->GetName()).find("convVtxYvsX") != string::npos ||
                       string(h->GetName()).find("convVtxRvsZ") != string::npos) {
                    h->Draw();
                    c0->Modified();
                    c0->Update(); 
                    string filename = prefix + "_" + *vecIt + "_" + h->GetName() + ".png";
                    while (filename.find("/") != string::npos) {
                        filename.replace(filename.find("/"), 1, "_");
                    }
                    while (filename.find(" ") != string::npos) {
                        filename.replace(filename.find(" "), 1, "_");
                    }
                    c0->SaveAs(filename.c_str());
                    outfile << "<a href=\"" << filename << "\"><img height=\"200\" width=\"324\" src=\"" << filename << "\"></a>\n";
                    continue;
                }

                bool is1D = true;
                
                if (obj->IsA()->InheritsFrom("TH2")) {
                    h->SetMarkerColor(4);
                    is1D = false;
                } 

                int firstBin = 1;
                int lastBin  = h->GetNbinsX();
                    
                if (h->GetEntries() == 0) {
                } else {
                    if (is1D) {
                        while (h->GetBinContent(firstBin) == 0. && firstBin < lastBin) ++firstBin;
                        while (h->GetBinContent(lastBin)  == 0. && firstBin < lastBin) --lastBin;
                    }


                    h->SetFillStyle(0);
                    h->SetLineColor(4);
                    h->SetLineWidth(2);
                    scale = 1./h->GetEntries();
                }

                if (vecIt->find("Efficiencies") == string::npos && 
                        !(obj->IsA()->InheritsFrom("TProfile"))) {
                }
                THStack * stack = new THStack("stack",h->GetTitle());
                stack->Add(h);



                stack->Draw("nostack");
                if (is1D) stack->GetXaxis()->SetRange(firstBin, lastBin);
                stack->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
                stack->GetYaxis()->SetTitle(h->GetYaxis()->GetTitle());
                stack->Draw("nostack");
                leg->Draw("same");
                c0->Modified();
                c0->Update(); 
                string filename = prefix + "_" + *vecIt + "_" + h->GetName() + ".png";
                while (filename.find("/") != string::npos) {
                    filename.replace(filename.find("/"), 1, "_");
                }
                while (filename.find(" ") != string::npos) {
                    filename.replace(filename.find(" "), 1, "_");
                }
                c0->SaveAs(filename.c_str());
                outfile << "<a href=\"" << filename << "\"><img height=\"200\" width=\"324\" src=\"" << filename << "\"></a>\n";
            }
        }

    }

    outfile << "</body>\n";
    outfile << "</html>\n";
    outfile.close();

    cout << "DONE!  HTML file created:" << endl;
    cout << ofname << endl;
}

EOF

root -b -q plotScript.C++\(\"$rootfile\"\)
mkdir -p $destdir/$shortname
/bin/mv ${shortname}_root.html $destdir/$shortname/index.html 
/bin/mv ${shortname}_root* $destdir/$shortname
