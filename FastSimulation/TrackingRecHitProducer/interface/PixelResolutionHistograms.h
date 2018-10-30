#ifndef FastSimulation_TrackingRecHitProducer_PixelResolutionHistograms_h
#define FastSimulation_TrackingRecHitProducer_PixelResolutionHistograms_h 1

class TFile;
class TH1F;
class TH2F;
class TAxis;
class RandomEngineAndDistribution;
class SimpleHistogramGenerator;

#include <memory>

/// #define COTBETA_HIST_MAX  30
/// #define COTALPHA_HIST_MAX 20
/// #define QBIN_HIST_MAX      4

static constexpr unsigned int COTBETA_HIST_MAX  = 30;
static constexpr unsigned int COTALPHA_HIST_MAX = 20;
static constexpr unsigned int QBIN_HIST_MAX     =  4;

class PixelResolutionHistograms {
public:

  //--- Constructor to use when generating resolution histograms.  
  //    We make empty histograms (which we own), but generator pointers 
  //    remain null.
  //
  PixelResolutionHistograms( std::string filename,    // ROOT file for histograms
			     std::string rootdir,     // Subdirectory in the file, "" if none
			     std::string descTitle,   // Descriptive title	     
			     unsigned int detType,     // Where we are... (&&& do we need this?)
			     double cotbetaBinWidth,   // cot(beta) : bin width,
			     double cotbetaLowEdge,    //           : low endpoint,
			     int    cotbetaBins,       //           : # of bins
			     double cotalphaBinWidth,  // cot(alpha): bin width, 
			     double cotalphaLowEdge,   //           : low endpoint,
			     int    cotalphaBins );    //           : # of bins
			   //int	qbinWidth,
			   //int	qbins )

  
  //--- Constructor to use when reading the histograms from a file (e.g. when 
  //    inside a running FastSim job).  We get the histograms from a
  //    ROOT file, and we do *not* own them.  But we do own the
  //    generators.
  //
  PixelResolutionHistograms( std::string filename,      // ROOT file for histograms
			     std::string rootdir = "",  // ROOT dir, "" if none
			     int   detType = -1,         // default: read from ROOT file.
			     bool  ignore_multi = false, // Forward Big is always single
			     bool  ignore_single = false,   // Edge does not need single pixels
			     bool  ignore_qBin = false  );  // qBin histograms not used right now (future expansion)


  //--- Destructor (virtual, just in case)
  virtual ~PixelResolutionHistograms();

  //--- Status after construction (esp.loading from file). Non-zero if there
  //    were problems.
  inline int status() { return status_ ; }

  //--- Fill one entry in one resolution histogram.  Use when making histograms.
  int Fill( double dx, double dy,     // the difference wrt true hit 
	    double cotalpha, double cotbeta,  // cotangent of local angles
	    int qbin,                 // Qbin = category for how much charge we have
	    int nxpix, int nypix );   // length of cluster along x,y (only care if ==1 or not)


  //--- Get generators, for resolution in X and Y.  Use in FastSim.
  const SimpleHistogramGenerator * getGeneratorX( double cotalpha,
						  double cotbeta, 
						  int qbin,
						  bool singlex );

  const SimpleHistogramGenerator * getGeneratorY( double cotalpha,
						  double cotbeta, 
						  int qbin,
						  bool singley );


 private:
  // Do we own the histograms, or not?
  bool weOwnHistograms_   ; 

  // Where we are.
  unsigned int detType_    ;  // 1 for barrel, 0 for forward  /// May not need this?

  // Resolution binning
  double cotbetaBinWidth_  ;
  double cotbetaLowEdge_   ;
  int	 cotbetaBins_	   ;
  double cotalphaBinWidth_ ;
  double cotalphaLowEdge_  ;
  int	 cotalphaBins_	   ;
  int	 qbinWidth_	   ;
  int	 qbins_  	   ;

  // The dummy histogram to hold the binning, and the two cached axes.
  TH2F * binningHisto_ ;
  TAxis * cotbetaAxis_  ;
  TAxis * cotalphaAxis_ ; 
  
  // Resolution histograms.  I (Petar) tried to dynamically allocate
  // these histograms, but all possible implementations were somewhat
  // complicated, which would make the code harder to understand,
  // debug, and thus support in the long term.  Since we are here only
  // booking pointers of histograms, we will instead book larger
  // matrices, and leave them partially empty.  But who cares -- the
  // wasted memory of a few hundred null pointers is negligible.
  //
  // The constructor will then fill only the first cotbetaBins_ x
  // cotalphaBins_ x qbinBins_ histograms in the matrix, and we'll
  // ignore the rest.
  //
  TH1F *  resMultiPixelXHist_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ][ QBIN_HIST_MAX ] ;
  TH1F * resSinglePixelXHist_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];
  TH1F *  resMultiPixelYHist_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ][ QBIN_HIST_MAX ];
  TH1F * resSinglePixelYHist_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];
  TH1F *            qbinHist_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];

  // File with histograms to load.
  std::unique_ptr<TFile> file_ ;
  
  // Status of loading.  Check if there were errors.
  int     status_ ;       

  // Identical binning and parameterization for FastSim generators.
  SimpleHistogramGenerator *  resMultiPixelXGen_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ][ QBIN_HIST_MAX ] ;
  SimpleHistogramGenerator * resSinglePixelXGen_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];
  SimpleHistogramGenerator *  resMultiPixelYGen_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ][ QBIN_HIST_MAX ];
  SimpleHistogramGenerator * resSinglePixelYGen_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];
  SimpleHistogramGenerator *            qbinGen_ [ COTBETA_HIST_MAX ][ COTALPHA_HIST_MAX ];
  
};
#endif
