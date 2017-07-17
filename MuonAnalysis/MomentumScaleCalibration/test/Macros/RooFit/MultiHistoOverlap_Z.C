#include <MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/MultiHistoOverlapAll_Z.C>

template <typename T> string separatebycommas(vector<T> v){
  if (v.size()==0) return "";
  stringstream s;
  s << v[0];
  for (unsigned int i = 1; i < v.size(); i++) s << "," << v[i];
  return s.str();
}
void MultiHistoOverlap_Z(){
  vector<string> filenames; vector<string> titles; vector<int> colors; vector<int> linestyles; vector<int> markerstyles;
  filenames.push_back("./BiasCheck.root"); titles.push_back("Data"); colors.push_back(1); linestyles.push_back(1); markerstyles.push_back(20);
  filenames.push_back("./BiasCheck_Reference.root"); titles.push_back("Reference"); colors.push_back(600); linestyles.push_back(1); markerstyles.push_back(1);

  TkAlStyle::legendheader = "";
  TkAlStyle::set(INTERNAL, NONE, "", "");

  MultiHistoOverlapAll_Z(separatebycommas(filenames), separatebycommas(titles), separatebycommas(colors), separatebycommas(linestyles), separatebycommas(markerstyles), "./plots", false);
}
