{

ifstream fpt;
fpt.open("runname.txt");
if (!fpt) { cout << "Error opening file ptranges.txt" << endl; assert(0);  }

string filename;
//Read in the cache file and store back to array
fpt >> filename;

TH1F htot("htot","Sum of all timeboxes",1000,0.,5000.);

TFile f(filename.c_str());
f.cd();

TH1F *htemp;

int count = 0;

for(int wheel=-2;wheel<3;wheel++){
  for(int sector=1;sector<13;sector++){
    for(int chamber=1;chamber<5;chamber++){
      for(int SL=1;SL<4;SL++){

	stringstream swheel; swheel << wheel;    
	stringstream sstation; sstation << chamber;
	stringstream ssector; ssector << sector; 
	stringstream ssuperLayer; ssuperLayer << SL;

	string histoname = "Ch_" + swheel.str() + "_" + sstation.str() + "_" + ssector.str() + "_SL" + ssuperLayer.str() + "_hTimeBox";

	cout << histoname.c_str() << endl;

	htemp = (TH1F*)f.Get(histoname.c_str());

	if(!htemp) continue;

	if(count == 0) htot = *htemp;
	else htot.Add(htemp);

      }
    }
  }
}

//f.Close();

TFile fout("htimeboxes.root","RECREATE");
fout.cd();
htot.Write();
fout.Close();

}
