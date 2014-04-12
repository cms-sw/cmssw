
void align(string path)
{
  gSystem->Load("AlignLib_C.so");

//   writeShifts(path);
//   writePars(path);

  AlignPlots a(path + "shifts");
  a.iter(0);
  a.iter(100);
  a.iters();
  AlignPlots b(path + "pars");
  b.iter(0);
  b.iter(100);
  b.iters();

//   if (!TFile((path + "histograms.root").c_str()).IsOpen()) return;
// 
//   string constraints[] = {"Det", "Pixel",
//     "TPBLadder", "TPBLayer", "TPBHalfBarrel", "TPBBarrel",
//     "TPEPanel", "TPEBlade", "TPEHalfDisk", "TPEHalfCylinder", "TPEEndcap"};
// 
//   for (int l = 0; l < 11; ++l)
//   {
//     writeSurvey(path, constraints[l]);
//     AlignPlots c(path + "survey_" + constraints[l]);
//     c.iters();
//   }
}

void align(string path, unsigned int subdet, int minHit = 0)
{
  gSystem->Load("AlignLib_C.so");

//   writeShifts(path);
//   writePars(path);

  vector<unsigned int> levels(1, subdet);

  AlignPlots a(path + "shifts", levels, minHit);
  a.iter(0);
  a.iter(100);
  a.iters();
  AlignPlots b(path + "pars", levels, minHit);
  b.iter(0);
  b.iter(100);
  b.iters();
}

void align()
{
  string scenarios[] = {"idealStrip", "10pbStrip", "realStrip"};
  string constraints[] = {"tracks", "survey", "merged"};

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      string path = scenarios[i] + '/' + constraints[j] + "/main/";
      align(path);
      align(path, 1);
      align(path, 2);
      align(path, 2, 500);
    }
}

/*
void align(TString path)
{
  gStyle->SetOptStat(1110);

  TFile fu(path + "IOUserVariables.root");
  TFile fs(path + "shifts.root");
  TTree* tu = (TTree*)fu.Get("T9_10");
  TTree* ts = (TNtupleD*)fs.Get("t10");
  ts->AddFriend(tu, "tu");

  const char* vars[] = {"x", "y", "z", "a", "b", "g"};

  TCanvas c("c", "c", 1200, 800);
  c.Divide(3, 2);

  for (int i = 0; i < 6; ++i)
  {
    c.cd(i + 1);
    ts->Draw(vars[i], "Nhit > 500");
  }

  c.SaveAs(path + "shifts_hit500+.png");
}

void align(string file, int iter, unsigned int subdet, unsigned int layer)
{
  TString vars[] = {'x', 'y', 'z', 'a', 'b', 'g'};

  char tree[] = {'t', '0' + iter, '\0'};

  TFile fin((file + ".root").c_str());
  TTree* t = (TTree*)fin.Get(tree);

  TCut cut1 = TString("subdetId(id) == ") + subdet;
  TCut cut2 = TString("layerNumber(id) == ") + layer;

  TCanvas c("c", "c", 1200, 800);
  c.Divide(3, 2);

  for (int i = 0; i < 6; ++i)
  {
    c.cd(i + 1);
    t->Draw(vars[i], cut1 && cut2);
  }

  TString out(file);
  out += iter; out += subdet == 3 ? "TIB" : "TOB";
  out += "Layer"; out += layer; out += ".png";
  c.SaveAs(out);
}

void align(string file, int iter)
{
  gSystem->Load("NumbersAndNames_C.so");

  for (int l = 1; l <= 4; ++l) align(file, iter, Namer::TIB, l);
  for (int l = 1; l <= 5; ++l) align(file, iter, Namer::TOB, l);
}

void align(string file)
{
  gSystem->Load("AlignLib_C.so");
  gSystem->Load("NumbersAndNames_C.so");

  for (int l = 1; l <= 4; ++l)
  {
    AlignPlots b("main/shifts", Namer::TIB, l); b.iters();
  }

  for (int l = 1; l <= 5; ++l)
  {
    AlignPlots b("main/shifts", Namer::TOB, l); b.iters();
  }
}
*/
