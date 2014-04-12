#ifndef __private_mlp_gen_h
#define __private_mlp_gen_h

#if defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#	define MLP_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef double dbl;
typedef double type_pat;

/* definition du reseau */
extern struct net_
{
  int Nlayer, *Nneur, Nweights;
  dbl ***Weights;
  dbl **vWeights;
  dbl **Deriv1, **Inn, **Outn, **Delta;
  int **T_func;
  int Rdwt, Debug;
} net_ MLP_HIDDEN;
#define NET net_   

/* apprentissage */
extern struct learn_
{
	int Nepoch, Meth, Nreset;
	dbl Tau,Norm,Decay,Lambda,Alambda;
	dbl eta, epsilon, delta;
	dbl ***Odw;
	dbl ***DeDw, ***ODeDw;
} learn_ MLP_HIDDEN;
#define LEARN learn_

extern struct pat_
{
	int Npat[2], Iponde, Nin, Nout;
	type_pat ***Rin, ***Rans, **Pond;
	type_pat **vRin; 
	dbl Ponds[10];
} pat_ MLP_HIDDEN;
#define PAT pat_

extern struct divers_
{
	int Dbin;
	int Ihess;
	int Norm, Stat;
	char Outf;
} divers_ MLP_HIDDEN;
#define DIVERS divers_	

extern struct stat_
{
	dbl *mean,*sigma;
} stat_ MLP_HIDDEN;
#define STAT stat_
	
extern int MessLang MLP_HIDDEN;
extern int OutputWeights MLP_HIDDEN;
extern int ExamplesMemory MLP_HIDDEN;
extern int WeightsMemory MLP_HIDDEN;
extern int PatMemory[2] MLP_HIDDEN;
extern int BFGSMemory MLP_HIDDEN;
extern int JacobianMemory MLP_HIDDEN;
extern int LearnMemory MLP_HIDDEN;
extern int NetMemory MLP_HIDDEN;
extern float MLPfitVersion MLP_HIDDEN;
extern dbl LastAlpha MLP_HIDDEN;
extern int NLineSearchFail MLP_HIDDEN;

extern dbl ***dir MLP_HIDDEN;
extern dbl *delta MLP_HIDDEN;
extern dbl **BFGSH MLP_HIDDEN;
extern dbl *Gamma MLP_HIDDEN;
extern dbl **JacobianMatrix MLP_HIDDEN;
extern int *ExamplesIndex MLP_HIDDEN;
extern dbl **Hessian MLP_HIDDEN;

extern void 	MLP_Out(type_pat *rrin, dbl *rrout) MLP_HIDDEN;
extern void 	MLP_Out2(type_pat *rrin) MLP_HIDDEN;
extern void 	MLP_Out_T(type_pat *rrin) MLP_HIDDEN;
extern dbl  	MLP_Test(int ifile, int regul) MLP_HIDDEN;
extern dbl 	MLP_Epoch(int iepoch, dbl *alpmin, int *ntest) MLP_HIDDEN;
extern int 	MLP_Train(int *ipat,dbl *err) MLP_HIDDEN;
extern dbl 	MLP_Stochastic() MLP_HIDDEN;
	
extern int 	StochStep() MLP_HIDDEN;

extern dbl 	DeDwNorm() MLP_HIDDEN;
extern dbl 	DeDwProd() MLP_HIDDEN;
extern void 	DeDwZero() MLP_HIDDEN;
extern void 	DeDwSaveZero() MLP_HIDDEN;
extern void 	DeDwScale(int Nexamples) MLP_HIDDEN;
extern void 	DeDwSave() MLP_HIDDEN;
extern int 	DeDwSum(type_pat *ans, dbl *out, int ipat) MLP_HIDDEN;

extern int 	SetTransFunc(int layer, int neuron, int func) MLP_HIDDEN;
extern void 	SetDefaultFuncs() MLP_HIDDEN;

extern void 	SteepestDir() MLP_HIDDEN;
extern void 	CGDir(dbl beta) MLP_HIDDEN;
extern dbl 	DerivDir() MLP_HIDDEN;
extern void 	GetGammaDelta() MLP_HIDDEN;
extern void 	BFGSdir(int Nweights) MLP_HIDDEN;
extern void 	InitBFGSH(int Nweights) MLP_HIDDEN;
extern int 	GetBFGSH(int Nweights) MLP_HIDDEN;

extern int 	LineSearch(dbl *alpmin, int *Ntest, dbl Err0) MLP_HIDDEN;
extern int 	DecreaseSearch(dbl *alpmin, int *Ntest, dbl Err0) MLP_HIDDEN;
extern void  	MLP_ResLin() MLP_HIDDEN;
extern void 	MLP_Line(dbl ***w0, dbl alpha) MLP_HIDDEN;
extern int	LineSearchHyb(dbl *alpmin, int *Ntest) MLP_HIDDEN;
extern void  	MLP_LineHyb(dbl ***w0, dbl alpha) MLP_HIDDEN;
extern int 	StochStepHyb() MLP_HIDDEN;
extern int	FixedStep(dbl alpha) MLP_HIDDEN;

extern void     EtaDecay() MLP_HIDDEN;
extern int 	ShuffleExamples(int n, int *index) MLP_HIDDEN;
extern double 	MLP_Rand(dbl min, dbl max) MLP_HIDDEN;
extern void	InitWeights() MLP_HIDDEN;
extern int	NormalizeInputs() MLP_HIDDEN;
extern int 	MLP_StatInputs(int Nexamples, int Ninputs, type_pat **inputs, 
		dbl *mean, dbl *sigma, dbl *minimum, dbl *maximum) MLP_HIDDEN;	
extern int 	MLP_PrintInputStat() MLP_HIDDEN;

extern int 	LoadWeights(char *filename, int *iepoch) MLP_HIDDEN;
extern int 	SaveWeights(char *filename, int iepoch) MLP_HIDDEN;

extern void 	SetLambda(double Wmax) MLP_HIDDEN;
extern void 	PrintWeights() MLP_HIDDEN;
extern int	ReadPatterns(char *filename, int ifile, 
		     int *inet, int *ilearn, int *iexamples) MLP_HIDDEN;
extern int 	CountLexemes(char *string) MLP_HIDDEN;
extern void 	getnLexemes(int n, char *s, char **ss) MLP_HIDDEN;
extern void 	getLexemes(char *s,char **ss) MLP_HIDDEN;
extern int      LearnAlloc() MLP_HIDDEN;
extern void 	LearnFree() MLP_HIDDEN;
extern int    	MLP_PrFFun(char *filename) MLP_HIDDEN;
extern int    	MLP_PrCFun(char *filename) MLP_HIDDEN;
extern int 	AllocPatterns(int ifile, int npat, int nin, int nout, int iadd) MLP_HIDDEN;
extern int 	FreePatterns(int ifile) MLP_HIDDEN;
extern void 	AllocWeights() MLP_HIDDEN;
extern void	FreeWeights() MLP_HIDDEN;
extern int 	AllocNetwork(int Nlayer, int *Neurons) MLP_HIDDEN;
extern void	FreeNetwork() MLP_HIDDEN;
extern int 	GetNetStructure(char *s, int *Nlayer, int *Nneur) MLP_HIDDEN;
extern int 	MLP_SetNet(int *nl, int *nn) MLP_HIDDEN;

extern void 	MLP_MM2rows(dbl* c, type_pat* a, dbl* b,
             	int Ni, int Nj, int Nk, int NaOffs, int NbOffs) MLP_HIDDEN;
extern void 	MLP_MatrixVector(dbl *M, type_pat *v, dbl *r, int n, 
				int m) MLP_HIDDEN;
extern void 	MLP_MatrixVectorBias(dbl *M, dbl *v, dbl *r, int n,
				 int m) MLP_HIDDEN;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __private_mlp_gen_h
