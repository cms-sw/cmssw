#ifndef Pythia6Random_h
#define Pythia6Random_h

#define m_length 6
#define r_length 100

class Pythia6Random
{

 public:

  Pythia6Random(int seed);
  ~Pythia6Random(void);
  int &mrpy(int i);
  double &rrpy(int i);
  void save(int i);
  void get(int i);
  
 private:
  
  void init(void);
  
  static int nDummy;
  static double dDummy;
  
  struct _pythia6random {
    int mrpy[m_length];
    double rrpy[r_length];
  };
  
  // This one points to the current PYDATR common block
  static struct _pythia6random *__pythia6random;
  
  // Those two contain the states of PYDATR after event generation
  // and after FamosSimHits.
  _pythia6random* myPythia6Random[2];
  
};
#endif
