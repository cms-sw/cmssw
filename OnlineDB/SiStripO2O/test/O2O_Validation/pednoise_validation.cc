#include "TFile.h"
#include "TIterator.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TString.h"
#include <iostream>
#include "TSystem.h"
#include "TObject.h"
#include <vector>
#include "TH1F.h"
#include "TCanvas.h"





//recursive function to retrieve Histos 
//Histos are filled to vector v_histo
void get_dir_structure(TString dir, TFile *file, std::vector<TString> &v_histo, std::vector<TString> &v_dir_save){
  //init variables
  TKey *file_key;
  TObject *file_object;
  TString histo_name="";
  TString dir_save_name="";
  TString dir_name[50000]={""};
  int count =0;

  //get data
  TDirectory *dir1=file->GetDirectory(dir);
  TIter file_iter(dir1->GetListOfKeys());

  //loop to get histos
  while((file_key)=((TKey*)file_iter())){
    file_object=file_key->ReadObj();
    dir_name[count]=file_object->GetName();
    dir_save_name=dir_name[count];
    dir_name[count]=dir+"/"+dir_name[count];
    if(  dir_name[count].Contains("module")
	 && ((dir_name[count].Contains("Profile_NoiseFromCondDB__det__")) || (dir_name[count].Contains("Profile_PedestalFromCondDB__det__")))
	 && file_object->InheritsFrom("TH1F"))
      {
       v_histo.push_back(dir_name[count]);
       v_dir_save.push_back(dir_save_name);
      }

    if(file_object->InheritsFrom("TDirectory")) {get_dir_structure(dir_name[count], file, v_histo, v_dir_save);}
            count++;
  }
  
      return;
}

//function searches for directory DQM in file and if there it calls get_dir_structure
void make_dir_tree(TString file_name, std::vector<TString> &v_dirs, std::vector<TString> &v_dir_save){
  TFile *f1=new TFile(file_name+".root", "read");
  std::cout<< " Opend file: " << file_name <<std::endl;
  std::cout<< " Reading directory structure"<<std::endl;

  TKey *file_key;
  TObject *file_object;
  TString histo_name="";
  TString dir_name[50000]={""};
  int count =0;
  TIter file_iter(f1->GetListOfKeys());
  while(file_key=(TKey*)file_iter()){
    file_object=file_key->ReadObj();
    dir_name[count]=file_object->GetName();
      count++;
  }

  get_dir_structure(dir_name[2],f1, v_dirs, v_dir_save);
  std::cout<< " Closing file: " << file_name <<std::endl;
  f1->Close();
  delete f1;
  return;
}


//compares if the number of histogramms and their pairwise sum(histo1-histo2)==0 on subdetector basis
//returns a vector that contains the non matching histos
std::vector<TString> compare_files(std::vector<TString> v_file1, std::vector<TString> v_file2, TString file1, TString file2, std::vector<TString> &v_dir_to_save, bool save_root_file){

  std::vector<TString> v_non_matching;
  uint32_t nr_diff_detids=0;
  
if(save_root_file){
  

  TH1F *histo_f1=new TH1F();
  TH1F *histo_f2=new TH1F();
  TFile *f1=new TFile(file1+".root", "read");
  TFile *f2=new TFile(file2+".root", "read");
  TFile *file_compare =new TFile(file1+"_"+file2+".root","update");
  bool TEC=true;
  bool TOB=true;
  bool TIB=true;
  bool TID=true;
  
  
  double xmax=0.0;
  double xmin=0.0;
  uint32_t nr_bins=0;
  uint32_t count=0;
  
  std::cout << "DetIds that don't match: "<< std::endl;
  
  std::vector<TString>::iterator iter_file1 = v_file1.begin();
  for(;iter_file1!=v_file1.end(); iter_file1++){
    if(iter_file1->Contains("TEC") && TEC){ std::cout << " Entering TEC!\n"; TEC=false;}
    if(iter_file1->Contains("TOB") && TOB){ std::cout << " Entering TOB!\n"; TOB=false;}
    if(iter_file1->Contains("TIB") && TIB){ std::cout << " Entering TIB!\n"; TIB=false;}
    if(iter_file1->Contains("TID") && TID){ std::cout << " Entering TID!\n"; TID=false;}

  
      std::vector<TString>::iterator iter_file2 = v_file2.begin();
      for(;iter_file2!=v_file2.end(); iter_file2++){
		if((iter_file1->Contains("TEC")) && !(iter_file2->Contains("TEC")))continue;
		if((iter_file1->Contains("TOB")) && !(iter_file2->Contains("TOB")))continue;
		if((iter_file1->Contains("TIB")) && !(iter_file2->Contains("TIB")))continue;
		if((iter_file1->Contains("TID")) && !(iter_file2->Contains("TID")))continue;

	//create diff histos
	if((*iter_file1).Contains(*iter_file2)){
      
	  f1->cd();
	  histo_f1=static_cast<TH1F*>(gDirectory->Get(*iter_file1));
	  xmax=histo_f1->GetXaxis()->GetXmax();
	  xmin=histo_f1->GetXaxis()->GetXmin();
	  nr_bins=histo_f1->GetNbinsX();
	  
	  f2->cd();
	  histo_f2=static_cast<TH1F*>(gDirectory->Get(*iter_file1));
	  	  
	  
	  TH1F *histo_compare=new TH1F((*iter_file1)+"_compare", *iter_file1+"_compare", nr_bins, xmin, xmax);
	  histo_compare->Add(histo_f1,1.);
	  histo_compare->Add(histo_f2,-1.);
	  
	  double integral=0.;
	  for(int i=0; i<=nr_bins; i++){
	    integral+=histo_compare->GetBinContent(i);
	  }
	  
	  if(integral!=0.){
	    file_compare->cd();
	    file_compare->mkdir(v_dir_to_save[count]);
	    file_compare->cd(v_dir_to_save[count]);
	    histo_f1->Write();
	    histo_f2->Write();
	    histo_compare->Write();
	    std::cout << " Found difference in detector:  "<< *iter_file1 << std::endl;
	    nr_diff_detids++;
	  }
	  delete histo_compare;
	  delete histo_f1;
	  delete histo_f2;
	  break;
	

	}
	
	if(!((*iter_file1).Contains(*iter_file2)) &&
	   (iter_file2 == (v_file2.end()-1))      ) {v_non_matching.push_back(*iter_file1);}
	

      }
      count++; 
	
	 
  }
 
  // diveded by 2 because ped+noise is counted
  std::cout << "Number of non matching DetIds: " << nr_diff_detids/2 << std::endl;

  std::cout << "Saving compare file" << std::endl;
  file_compare->cd();
  file_compare->Write();
   file_compare->Close();
   delete file_compare;
  std::cout << "Compare file closed!" << std::endl;
  f1->cd();
  f1->Close();
  delete f1;
  std::cout << "f1 file closed!" << std::endl;
 f2->cd();
  f2->Close();
  delete f2;
 std::cout << "f2 file closed!" << std::endl;
 std::cout << "Closed all files " << std::endl;
 
 }

 if(!save_root_file){
   TFile *f1=new TFile(file1+".root", "read");
    TFile *f2=new TFile(file2+".root", "read");
     bool TEC=true;
    bool TOB=true;
    bool TIB=true;
    bool TID=true;
 

    std::vector<TString>::iterator iter_file1 = v_file1.begin();
      for(;iter_file1!=v_file1.end(); iter_file1++){
	 if(iter_file1->Contains("TEC") && TEC){ std::cout << " Entering TEC!\n"; TEC=false;}
	 if(iter_file1->Contains("TOB") && TOB){ std::cout << " Entering TOB!\n"; TOB=false;}
	 if(iter_file1->Contains("TIB") && TIB){ std::cout << " Entering TIB!\n"; TIB=false;}
	 if(iter_file1->Contains("TID") && TID){ std::cout << " Entering TID!\n"; TID=false;}
	 std::vector<TString>::iterator iter_file2 = v_file2.begin();
	 for(;iter_file2!=v_file2.end(); iter_file2++){
	 	if((iter_file1->Contains("TEC")) && !(iter_file2->Contains("TEC")))continue;
		if((iter_file1->Contains("TOB")) && !(iter_file2->Contains("TOB")))continue;
		if((iter_file1->Contains("TIB")) && !(iter_file2->Contains("TIB")))continue;
		if((iter_file1->Contains("TID")) && !(iter_file2->Contains("TID")))continue;
		if(((*iter_file1).Contains(*iter_file2))) break;
 		if(!((*iter_file1).Contains(*iter_file2)) &&
 	           (iter_file2 == (v_file2.end()-1))      ) {v_non_matching.push_back(*iter_file1);}

       }
    }
      f1->cd();
      f1->Close();
      delete f1;
      std::cout << "f1 file closed!" << std::endl;
      f2->cd();
      f2->Close();
      delete f2;
      std::cout << "f2 file closed!" << std::endl;
      std::cout << "Closed all files " << std::endl;
 }
 
 

return v_non_matching;
}

void print_missing(std::vector<TString> v_mis_mod, TString str1){
  uint32_t count=0;
  for(uint32_t i=0;i<v_mis_mod.size();i++){
    std::cout << "Modules that are missing in file: " << str1 << " : " << v_mis_mod[i] << std::endl;
    count ++;
  }
  
  std::cout << "Nr of missing modules: " << count << std::endl;
  return;
}


int main(int argc, char *argv[]){
 if(argc !=5){
    std::cout << "Programm assumes the following parameters:\n" 
	      << "validata_noise_ped\t"
	      << "runnr 1\t"
	      << "runnr 2\t"
	      << "tag_old\t excluding trailing _"
	      << "tag_new\t excluding trailing _"
	      << "\n\n";
    return 0;
  }

 TString argv1=argv[1];
 TString argv2=argv[2];
 TString argv3=argv[3];
 TString argv4=argv[4];
 
 TString file_name1=argv3+"_"+argv1;
 TString file_name2=argv4+"_"+argv2;

 std::vector<TString> v_histos_file1;
 std::vector<TString> v_histos_file2;
 std::vector<TString> v_missing_modules_file1;
 std::vector<TString> v_missing_modules_file2;
 std::vector<TString> v_modules_to_save;

 make_dir_tree(file_name1, v_histos_file1, v_modules_to_save);
 make_dir_tree(file_name2, v_histos_file2, v_modules_to_save);

 std::cout << "Enter first compare files:\n";
 v_missing_modules_file1=compare_files(v_histos_file1, v_histos_file2, file_name1, file_name2, v_modules_to_save, true);
 std::cout << "Enter first print_missing:\n";
 print_missing(v_missing_modules_file1, file_name1);

 std::cout << "Enter second compare files:\n";
 v_missing_modules_file2=compare_files(v_histos_file2, v_histos_file1, file_name2, file_name1, v_modules_to_save, false);
 std::cout << "Enter second print_missing:\n";
 print_missing(v_missing_modules_file2, file_name2);


 

   return 0;


}
