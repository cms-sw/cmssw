using namespace std;
#include <TH1D.h>
#include <TAxis.h>
#include <TMath.h>
#include "EffPullCalculator.hh"

EffPullcalculator::EffPullcalculator(TH1D* pathHisto1_v, TH1D* pathHisto2_v, vector<TH1D*> sortedHisto1_v, vector<TH1D*> sortedHisto2_v, string error_v) {
  pathHisto.push_back(pathHisto1_v);
  pathHisto.push_back(pathHisto2_v);
  sortedHisto1 = sortedHisto1_v;
  sortedHisto2 = sortedHisto2_v;
  error = error_v;
}

void EffPullcalculator::CalculatePulls() {

  if(pathHisto[0]->GetNbinsX() != pathHisto[1]->GetNbinsX()) {
    lines.push_back("--------------------------------------------------------------------------------");
    lines.push_back("WARNING in EffPullcalculator::CalculatePulls: the vectors to compare ");
    lines.push_back("                                              have different size."   );
    lines.push_back("                                              Some HLT path was added or removed");
    lines.push_back(" ");
  }

  TAxis* axis1 = pathHisto[0]->GetXaxis();
  TAxis* axis2 = pathHisto[1]->GetXaxis();
  double nEv1 = 0;
  double nEv2 = 0;

  for(int i1=1; i1 <= pathHisto[0]->GetNbinsX(); i1++) {
    string label1 = axis1->GetBinLabel(i1);
    if(label1 == "Total") nEv1 = pathHisto[0]->GetBinContent(i1);
    for(int i2=1; i2 <= pathHisto[1]->GetNbinsX(); i2++) {
      string label2 = axis2->GetBinLabel(i2);
      if(label2 == "Total") nEv2 = pathHisto[1]->GetBinContent(i2);
      if(label1 == label2) { // we are comparing the same path
	eff1.push_back(pathHisto[0]->GetBinContent(i1));
	eff2.push_back(pathHisto[1]->GetBinContent(i2));
	name.push_back(label1);
      }
    }
  }

  for(int i = 0; i< int(eff1.size()); i++) {
    eff1[i] *= (nEv1 == 0 ? 0. : 1./nEv1);
    eff2[i] *= (nEv2 == 0 ? 0. : 1./nEv2);
     err_eff1.push_back(sqrt(eff1[i]*(1.-eff1[i])/nEv1));
     err_eff2.push_back(sqrt(eff2[i]*(1.-eff2[i])/nEv2));
//     err_eff1.push_back(sqrt(eff1[i]/nEv1));
//     err_eff2.push_back(sqrt(eff2[i]/nEv2));
  }

  // create 10x2 histograms with efficiencies
  int nbin = 10;//eff1.size()/8+1;
  vector<int> nbins; nbins.resize(20);
  nbins[0]=nbin;
  nbins[1]=nbin;
  nbins[10]=nbin;
  nbins[11]=nbin;
  for(int i=2; i<10; ++i)
    {
      if(i-2<int(sortedHisto1.size()))
	{
	  nbins[i] = sortedHisto1[i-2]->GetNbinsX();
	  nbins[i+10] = sortedHisto1[i-2]->GetNbinsX();
	}
      else
	{
	  nbins[i]=0;
	  nbins[i+10]=0;
	}
    }
  for(int iplot=0; iplot<20;iplot++) {
    char name[256];
    sprintf(name,"%s_eff_%i",pathHisto[0]->GetName(),iplot);
    effhisto.push_back(new TH1D(name, "Trigger Efficiency", nbins[iplot], 0., double(nbins[iplot])));
  }
  for(int i=2; i<10 && i-2<int(sortedHisto1.size()); ++i)
    {
      effhisto[i]->SetTitle(sortedHisto1[i-2]->GetTitle());
      effhisto[i+10]->SetTitle(sortedHisto1[i-2]->GetTitle());
    }

  for(int k=0; k<int(sortedHisto1.size()); ++k)
    {
      int i=0, j=0;
      cout<<"111\n";
      for(i=2; i<=sortedHisto1[k]->GetNbinsX(); ++i)
	{
	  if(std::abs(sortedHisto1[k]->GetBinContent(i) - sortedHisto2[k]->GetBinContent(i)) > std::abs(sortedHisto1[k]->GetBinContent(i-1)-sortedHisto2[k]->GetBinContent(i-1)))
	    {
	      double content1=sortedHisto1[k]->GetBinContent(i), content2=sortedHisto2[k]->GetBinContent(i), error1=sortedHisto1[k]->GetBinError(i), error2=sortedHisto2[k]->GetBinError(i);
	      char name1[256], name2[256], char_name[256];
	      sprintf(name1,"%s",sortedHisto1[k]->GetXaxis()->GetBinLabel(i));
	      sprintf(name2,"%s",sortedHisto2[k]->GetXaxis()->GetBinLabel(i));
	      for(j=i-1; j>=1; --j)
		{
		  if(std::abs(content1-content2) > std::abs(sortedHisto1[k]->GetBinContent(j) - sortedHisto2[k]->GetBinContent(j)))
		    {
		      sortedHisto1[k]->SetBinContent(j+1, sortedHisto1[k]->GetBinContent(j));
		      sortedHisto1[k]->SetBinError(j+1, sortedHisto1[k]->GetBinError(j));
		      sprintf(char_name,"%s",sortedHisto1[k]->GetXaxis()->GetBinLabel(j));
		      sortedHisto1[k]->GetXaxis()->SetBinLabel(j+1, char_name);
		      sortedHisto2[k]->SetBinContent(j+1, sortedHisto2[k]->GetBinContent(j));
		      sortedHisto2[k]->SetBinError(j+1, sortedHisto2[k]->GetBinError(j));
		      sprintf(char_name,"%s",sortedHisto2[k]->GetXaxis()->GetBinLabel(j));
		      sortedHisto2[k]->GetXaxis()->SetBinLabel(j+1, char_name);
		    }
		  else
		    break;
		}
	      sortedHisto1[k]->SetBinContent(j+1, content1);
	      sortedHisto1[k]->SetBinError(j+1, error1);
	      sortedHisto1[k]->GetXaxis()->SetBinLabel(j+1, name1);
	      sortedHisto2[k]->SetBinContent(j+1, content2);
	      sortedHisto2[k]->SetBinError(j+1, error2);
	      sortedHisto2[k]->GetXaxis()->SetBinLabel(j+1, name2);
	    }
	}
    }

  // set plot style
  for(int iplot=0; iplot<10;iplot++) {
    effhisto[iplot]->GetXaxis()->SetLabelSize(0.03);
    effhisto[iplot+10]->GetXaxis()->SetLabelSize(0.03);
    effhisto[iplot]->SetStats(0);
    effhisto[iplot+10]->SetStats(0);
    effhisto[iplot]->SetYTitle("Efficiency");
    effhisto[iplot+10]->SetYTitle("Efficiency");
    effhisto[iplot]->SetFillColor(38);
    effhisto[iplot+10]->SetFillColor(28);
    effhisto[iplot]->SetBarWidth(0.4*effhisto[iplot]->GetXaxis()->GetBinWidth(1));
    effhisto[iplot]->SetBarOffset(0.1*effhisto[iplot]->GetXaxis()->GetBinWidth(1));
    effhisto[iplot+10]->SetBarWidth(0.4*effhisto[iplot+10]->GetXaxis()->GetBinWidth(1));
    effhisto[iplot+10]->SetBarOffset(0.5*effhisto[iplot+10]->GetXaxis()->GetBinWidth(1));
  }

  for(int i=2; i<10;++i)
    {
      if(i-2<int(sortedHisto1.size()))
	for(int j=0; j<nbins[i]; ++j)
	  {
	    effhisto[i]->SetBinContent(j+1,sortedHisto1[i-2]->GetBinContent(j+1));
	    effhisto[i]->SetBinError(j+1,sortedHisto1[i-2]->GetBinError(j+1));
	    effhisto[i]->GetXaxis()->SetBinLabel(j+1,sortedHisto1[i-2]->GetXaxis()->GetBinLabel(j+1));
	    effhisto[i+10]->SetBinContent(j+1,sortedHisto2[i-2]->GetBinContent(j+1));
	    effhisto[i+10]->SetBinError(j+1,sortedHisto2[i-2]->GetBinError(j+1));
	    effhisto[i+10]->GetXaxis()->SetBinLabel(j+1,sortedHisto2[i-2]->GetXaxis()->GetBinLabel(j+1));
	  }
    }
  
  // Histograms with largest efficiencies
  vector<int> SortedEff = SortVec(eff2);

  int nb=1;
  for(int i=0; i< int(SortedEff.size()); i++) {
    if(eff1[SortedEff[i]] != 0. && name[SortedEff[i]] != "Total") {
      effhisto[0]->SetBinContent(nb,eff1[SortedEff[i]]);
      effhisto[0]->SetBinError(nb,err_eff1[SortedEff[i]]);
      effhisto[10]->SetBinContent(nb,eff2[SortedEff[i]]);
      effhisto[10]->SetBinError(nb,err_eff2[SortedEff[i]]);
      effhisto[0]->GetXaxis()->SetBinLabel(nb,name[SortedEff[i]].c_str());
      effhisto[10]->GetXaxis()->SetBinLabel(nb,name[SortedEff[i]].c_str());
      nb++;
    }
    if(nb == effhisto[0]->GetXaxis()->GetNbins()+1) break;
  }

  // Histograms with largest differences in efficiencies
  vector<double> deltaeff;
  for(int i=0; i< int(eff1.size()); i++) 
    deltaeff.push_back(fabs(eff1[i]-eff2[i]));
  SortedEff = SortVec(deltaeff);
  nb=1;
  for(int i=0; i< int(SortedEff.size()); i++) {
    if(eff2[SortedEff[i]] != 0. && eff1[SortedEff[i]] != 0.) {
      effhisto[1]->SetBinContent(nb,eff1[SortedEff[i]]);
      effhisto[1]->SetBinError(nb,err_eff1[SortedEff[i]]);
      effhisto[11]->SetBinContent(nb,eff2[SortedEff[i]]);
      effhisto[11]->SetBinError(nb,err_eff2[SortedEff[i]]);
      effhisto[1]->GetXaxis()->SetBinLabel(nb,name[SortedEff[i]].c_str());
      effhisto[11]->GetXaxis()->SetBinLabel(nb,name[SortedEff[i]].c_str());
      nb++;
    }
    if(nb == effhisto[1]->GetXaxis()->GetNbins()+1) break;
  }
  
  // pull histogram
  pullDist = new TH1D(string(string(pathHisto[0]->GetName())+"_pullDist").c_str(), 
		      "Efficiency pulls", 20, -10., 10.);
  TAxis* pullXaxis = pullDist->GetXaxis();
  pullXaxis->SetTitle("Pull of trigger efficiency");
  
  //  residual histogram
  resHisto = new TH1D(string(string(pathHisto[0]->GetName())+"_resHisto").c_str(), 
		      "Efficiency residuals", eff1.size(), 0., eff1.size()*1.);
  TAxis* resXaxis = resHisto->GetXaxis();
  resXaxis->SetTitle("");
  TAxis* resYaxis = resHisto->GetYaxis();
  resYaxis->SetTitle("Efficiency Residual");
  
  for(int i = 0; i< int(eff1.size()); i++) {
    if(eff1[i] == 0. && eff2[i] != 0) {
      lines.push_back("---------------------------------------------------------------------------------------------");
      lines.push_back("WARNING: " + name[i] + " Trigger Path was masked and now it is used");
      lines.push_back(" ");
    } else if(eff2[i] == 0. && eff1[i] != 0) {
      lines.push_back("---------------------------------------------------------------------------------------------");
      lines.push_back("WARNING: " + name[i] + " Trigger Path was used and now it is masked");
      lines.push_back(" ");
    } else {
      double err = (error == "uncorrelated" ? 
		    sqrt(std::pow(err_eff1[i],2.)+ std::pow(err_eff2[i],2.)) :
		    err_eff1[i]-err_eff2[i]);
      double pull = (eff1[i]-eff2[i])/err;
      if(fabs(pull) > 3.) {
	lines.push_back("---------------------------------------------------------------------------------------------");
	lines.push_back("WARNING: A discrepancy bigger than 3 sigmas was found for " + name[i] + " Trigger Path");
	lines.push_back(" ");
      }
      
      resXaxis->SetBinLabel(i+1,name[i].c_str());
      pullDist->Fill(pull);
      resHisto->SetBinContent(i+1,eff1[i]-eff2[i]);
      resHisto->SetBinError(i+1,err);
    }
  }
}

double EffPullcalculator::GetEff(string label, int ind) {
  double eff = 0.;
  if(ind<2) {
    TAxis* axis = pathHisto[ind]->GetXaxis();
    for(int i=1; i<=pathHisto[ind]->GetNbinsX(); i++) {
      if(axis->GetBinLabel(i) == label) {
	// last bin is total
	eff = pathHisto[ind]->GetBinContent(i)/pathHisto[ind]->GetBinContent(pathHisto[ind]->GetNbinsX());
      }
    }
  }
  return eff;
}

void EffPullcalculator::WriteLogFile(string namefile) {
  FILE*  f=fopen(namefile.c_str(),"w");
  for(int j = 0; j< int(lines.size()); j++) 
    fprintf(f,"%s\n",lines[j].c_str());
  fclose(f);
}

vector<int> EffPullcalculator::SortVec(vector<double> eff) {
  vector<double> eff_tmp = eff;
  vector<double> eff_tmp2 = eff;
  vector<int> indexes;

  std::sort(eff_tmp.begin(), eff_tmp.end());

  for(int i=0; i< int(eff_tmp.size()); i++) 
    for(int j=0; j< int(eff.size()); j++) 
      if(eff_tmp2[j] == eff_tmp[eff_tmp.size()-1-i]) {
	indexes.push_back(j);
	eff_tmp2[j] = 1.1; // to ignore it the next round
	//	eff_tmp2.remove(i);
	j=0;
	i++;
      }
  
  return indexes;
}

void EffPullcalculator::AddGoldenPath(string name) {
  goldenpaths.push_back(name);
}

void EffPullcalculator::PrintTwikiTable(string filename) {

  FILE* f=fopen(filename.c_str(),"w");

  for(int j=0; j< int(goldenpaths.size()); j++) {
    for(int i=0; i < int(name.size()); i++) {
      if(name[i] == goldenpaths[j]) {
	fprintf(f,"%s|%i +/- %i|%i +/- %i |\n",name[i].c_str(),eff1[i],err_eff1[i],eff2[i],err_eff2[i]);
      }
    }
  }
  fclose(f);
}

double EffPullcalculator::abs(double value)
{
  if(value<0)
    return -1.0*value;
  else
    return value;
}
