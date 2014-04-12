#ifndef JETMETCOMP_HH
#define JETMETCOMP_HH

class JetMETComp {
public:
  JetMETComp(map<string,double>);
  ~JetMETComp(){};
  
  void MakePlot(string name);
  void WriteFile();

private:
  map<string,double> compatibilities;
  vector<string> lines;
  
};

#endif
