#ifndef __private_mlp_sigmoide_h
#define __private_mlp_sigmoide_h

#if defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#	define MLP_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

double 	MLP_Sigmoide(double x) MLP_HIDDEN;
void 	MLP_vSigmoide(double *x, int n) MLP_HIDDEN;
void 	MLP_vSigmoideDeriv(double *x, double *dy, int n) MLP_HIDDEN;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __private_mlp_sigmoide_h
