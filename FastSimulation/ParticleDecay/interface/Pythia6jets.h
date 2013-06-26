#ifndef Pythia6jets_h
#define Pythia6jets_h

#define _length 5
#define _depth 4000

class Pythia6jets
{
public:
	Pythia6jets(void);
	~Pythia6jets(void);
	int &n(void);
	int &npad(void);
	int &k(int i,int j);
	double &p(int i,int j);
	double &v(int i,int j);

private:
	void init(void);

	static int nDummy;
	static double dDummy;

	struct _pythia6jets {
		int n;
		int npad;
		int k[_length][_depth];
		double p[_length][_depth];
		double v[_length][_depth];
	};

	static struct _pythia6jets *__pythia6jets;
};
#endif
