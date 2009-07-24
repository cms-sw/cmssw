/**
	@file					HLTDatasets.h
	@author				Sue Ann Koay (sakoay@cern.ch)

  @mainpage

  HLTDatasets is the main class that the user should pay attention to. Its detailed 
  documentation includes a recipe for how to install and run it within some event loop.

  The book-keeping is done this way:
    - A HLTDatasets object contains several SampleDiagnostic objects, one for each sample 
      (MinBias, QCD, etc.) that the user specifies to run over. All the SampleDiagnostics
      have the same functionality, the division is just to be able to identify the different
      contributions from different samples.
    - A SampleDiagnostics object is made up of several Dataset objects. The Dataset list
      is set up by HLTDatasets from the input specifications file that the user has to 
      provide (to HLTDatasets) at construction time.
    - A Dataset is an OR of Trigger's. It knows how to check if an event is contained in it.
      It also keeps track of the number of events that it contains, and how many events it
      has distinct from some other Dataset.
*/


#ifndef __HLTDATASETS_H__
#define __HLTDATASETS_H__

#include <fstream>
#include <vector>
#include <algorithm>

#include <TH2.h>
#include <TString.h>
#include <TStopwatch.h>


//=============================================================================
// Supporting Classes
//=============================================================================

/**
  Contains information for a particular trigger. The name is used to specify the
  trigger (from the datasets configuration file etc.), the index is internally
  used to locate the trigger bit.
*/
class Trigger {
public:
  //.. Data ...................................................................
  TString   name;     ///< Name of the trigger path.
  Int_t     index;    ///< Internal index for trigger bit access.

  //.. Functions ..............................................................
  /// Constructs a Trigger. setup() must be called to properly associate the index.
  Trigger ( const Char_t    triggerName[] = ""    ///< Name of the trigger path.
          , Int_t           triggerIndex  = -1    ///< Internal index for trigger bit access.
          )
    : name  (triggerName)
    , index (triggerIndex)
  { }
};

//-----------------------------------------------------------------------------

/**
  A Dataset is defined by a bunch of triggers. An event is accepted if any one
  of those triggers fire; this decision is stored by the checkEvent() member function.
  A Dataset also has storage for the number of events that it can contribute to
  some other dataset if they were merged. 
*/
class Dataset : public std::vector<Trigger> {
public:
  using std::vector<Trigger>::at;
  using std::vector<Trigger>::size;
  //.. Data ...................................................................
  TString               name;             ///< Name used to identify the dataset.
  Bool_t                isNewTrigger;     ///< True if this "dataset" is actually just a single new trigger to be considered, false otherwise.
  UInt_t                numEventsPassed;  ///< Cumulative number of events that has fired this dataset, so far.
  Double_t              rate;             ///< Storage for the rate of firing this dataset, as computed by computeRate().
  Double_t              rateUncertainty2; ///< Storage for the rate uncertainty squared, as computed by computeRate().
  std::vector<UInt_t>   numEventsAdded;   ///< Number of events that fired this dataset, but is @e not present in the other (indexed) dataset in this list.
  std::vector<Double_t> addedRate;        ///< Rate of events that fired this dataset, but is @e not present in the other (indexed) dataset in this list.
  std::vector<Double_t> addedUncertainty2;///< Squared uncertainty of rate of events that fired this dataset, but is @e not present in the other (indexed) dataset in this list.
  std::vector<UInt_t>   datasetIndices;   ///< Indices of the datasets being compared to for numEventsAdded.
  Bool_t                pass;             ///< True if the current event fires one or more triggers in this dataset, false otherwise.

  //.. Functions ..............................................................
  /// Constructs a dataset with the given name.
  Dataset ( const Char_t  datasetName[] = ""      ///< Name for identifying this dataset.
          , Bool_t        isNewTrigger  = kFALSE  ///< True if this "dataset" is actually just a single new trigger to be considered, false otherwise.
          )
    : name              (datasetName)
    , isNewTrigger      (isNewTrigger)
    , numEventsPassed   (0)
    , rate              (0)
    , rateUncertainty2  (0)
  { }
  /**
    Sets up storage for this dataset, given the list of datasets in the scenario.
  */
  void setup(const std::vector<Dataset>&    datasets
            );

  /// Adds the rate information of another dataset to this one. The event counts are @e not added.
  Dataset&  operator+=(const Dataset& addend    ///< Dataset whose rate information is ot be added to this one.
                      );
  /// Checks the name of the dataset.
  Bool_t    operator==(const Char_t datasetName[]) const  { return name == datasetName; }
  /// Checks the name of the dataset.
  Bool_t    operator!=(const Char_t datasetName[]) const  { return name != datasetName; }
  /// Checks the name of the dataset.
  Bool_t    operator==(const TString& datasetName) const  { return name == datasetName; }
  /// Checks the name of the dataset.
  Bool_t    operator!=(const TString& datasetName) const  { return name != datasetName; }

  /**
    Checks whether or not the current event (as specified by some number of trigger bits) fires
    one or more triggers in this dataset. The decision is stored in the pass data member and also
    returned.
  */
  Bool_t checkEvent( const std::vector<Int_t>&   triggerBit    ///< The list of trigger bits for the current event.
                   );
  /**
    Checks how much a dataset would contribute to some other dataset, if the two are to be merged.
    The results are stored in the numEventsAdded data member.
  */
  void checkAddition(const std::vector<Dataset>&  datasets    ///< All the datasets in the scenario.
                    );

  /// Computes and stores the rate, using the stored number of events. 
  void computeRate( Double_t  collisionRate       ///< Rate of bunch crossings, for onverting numbers of events into rates.
                  , Double_t  mu                  ///< bunchCrossingTime * crossSection * instantaneousLuminosity * maxFilledBunches / nFilledBunches
                  , UInt_t    numProcessedEvents  ///< Number of events processed for this sample, to be used for the trigger efficiency calculations.
                  );

  /**
    Prints to the given output stream:
      - A Latex table of the contribution of this dataset to the other datasets (numEventsAdded).
    Make sure to call computeRate() first, if you prefer sensible output.
  */
  void report ( std::ofstream&                output                    ///< Stream to output to. Note that this just produces a table, not a fully compilable Latex.
              , const std::vector<Dataset>&   datasets                  ///< All the datasets in the scenario.
              , const Char_t*                 errata            = 0     ///< An additional description to be output at the end of the caption of the table, if desired.
              , const Int_t                   significantDigits = 3     ///< Number of significant digits to report percentages in.
              ) const;
};



//-----------------------------------------------------------------------------


/// Classification of the relevance of a particular sample.
enum  SampleCategory  { RATE_SAMPLE       ///< These are samples that drive the overall trigger rate, i.e. typically minbias/QCD.
                      , PHYSICS_SAMPLE    ///< These are samples that are of physics interest, e.g. top, W, Z, etc.
                      };

/// Container for per-sample diagnostics information, as used by HLTDatasets.
/**
  This class provides a container for the information used by HLTDatasets. There should be one 
  per sample because of the different cross-sections used to convert each sample efficiencies 
  into rates; moreover it allows for outputting diagnostics that are specific to a particular
  sample (instead of just one overall). A SampleDiagnostics object should be updated for each 
  event in the loop over the sample by calling the fill() member function. At the end of the 
  loop, it can be told to output diagnostics by calling the report() member function.
*/
class SampleDiagnostics : public std::vector<Dataset>
{
public:
  using std::vector<Dataset>::at;
  using std::vector<Dataset>::size;
  //.. Data ...................................................................
  TString                             name;                     ///< Name of the sample, for display purposes.
  SampleCategory                      typeOfSample;             ///< The type of sample (see the SampleCategory documentation), for presentation purposes.
  std::vector<std::vector<UInt_t> >   commonEvents;             ///< Number of events that commonly fired both x- and y-indexed datasets in this list. The last row for every x is the number of events in x and @e any y.
  std::vector<std::vector<Double_t> > commonRates;              ///< Rates of events common to both x- and y-indexed datasets in this list.
  std::vector<std::vector<Double_t> > commonRateUncertainties2; ///< Squared uncertainties on rates of events common to both x- and y-indexed datasets in this list.
  UInt_t                              numPassedEvents;          ///< Total number of events that passes any non-isNewTrigger dataset.
  Double_t                            passedRate;               ///< Total rate of events that passes any non-isNewTrigger dataset.
  Double_t                            passedRateUncertainty2;   ///< Uncertainty squared of rate of events that passes any non-isNewTrigger dataset.
  UInt_t                              numProcessedEvents;       ///< Number of events processed for this sample, to be used for the trigger efficiency calculations.
  UInt_t                              numConstituentSamples;    ///< Number of samples aggregated into this one, if relevant.
protected:
  Int_t                               firstNewTrigger;          ///< Index of the first dataset that is marked as isNewTrigger. For presentation purposes.
  TStopwatch*                         timer;                    ///< For bench-marking the execution, if you care.


public:
  //.. Functions ..............................................................
  /// Creates a SampleDiagnostics object for a particular sample. 
  SampleDiagnostics( const Char_t     sampleName[]  = ""              ///< Name of the sample, for display purposes.
                   , TStopwatch*      timer         = 0               ///< For bench-marking the execution, if you care.
                   , SampleCategory   typeOfSample  = PHYSICS_SAMPLE  ///< The type of sample (see the SampleCategory documentation), for presentation purposes.
                   )
    : name                  (sampleName)
    , typeOfSample          (typeOfSample)
    , numPassedEvents       (0)
    , passedRate            (0)
    , passedRateUncertainty2(0)
    , numProcessedEvents    (0)
    , numConstituentSamples (1)
    , firstNewTrigger       (-1)
    , timer                 (timer)
  { }
  /// Sets up the storage according to the number of datasets.
  void setup();

  /// Adds the rate information of another sample to this one. The event counts are @e not added.
  SampleDiagnostics&  operator+=(const SampleDiagnostics& addend    ///< SampleDiagnostics whose rate information is ot be added to this one.
                                );

  /**
    Records the dataset firing pattern of the current event (as specified by some number of 
    trigger bits). This should be called as part of the event loop. At the end of the loop, 
    computeRates() should be called to convert event counts into rates. Then report() can be
    called.
  */
  void fill( const std::vector<Int_t>&  triggerBit    ///< The list of trigger bits for the current event.
           );

  /**
    Computes and stores rates, using the stored number of events. Important to call at the end
    of the event loop, before the diagnostic output functions.
  */
  void computeRate( Double_t  collisionRate       ///< Rate of bunch crossings, for onverting numbers of events into rates.
                  , Double_t  mu                  ///< bunchCrossingTime * crossSection * instantaneousLuminosity * maxFilledBunches / nFilledBunches
                  );

  /**
    TObject::Write()s to the currently open file correlation plots [RateIn(X AND Y) / RateIn(X)], 
    where X and Y are either new triggers or primary datasets, as seen on the label on the @a x 
    and @a y axes respectively. The non-normalized versions RateIn(X AND Y) are stored in the 
    "unnormalized" sub-directory.
    @note   Make sure to call computeRate() first, if you prefer sensible output.
  */
  void  write() const;

  /**
    Makes the following diagnostics for this sample:
      - Prints to the given file a Latex table of the contribution of all datasets marked 
        as isNewTrigger, to the other datasets.
      - TObject::Write()s to the currently open file correlation plots 
        [RateIn(X AND Y) / RateIn(X)], where X and Y are either new triggers or
        primary datasets, as seen on the label on the x and y axes respectively. 
        The non-normalized versions RateIn(X AND Y) are stored in the "unnormalized"
        sub-directory.
    @note   Make sure to call computeRate() first, if you prefer sensible output.
  */
  void  report( TString           tablesPrefix  ///< Path and name of file to output the tables to. @c tablesPrefix.tex is created by this function, and @c tablesPrefix.pdf made by running latex.
              , const Char_t*     errata = 0    ///< An additional description to be output at the end of the caption of the table, if desired.
              ) const;
};



//=============================================================================
// User Interface
//=============================================================================

/// Diagnostic plots for primary dataset definitions, including the impact of adding new triggers.
/**
  This class provides a collection of information useful for deciding what triggers to add to an
  existing core set, and which primary dataset(s) such new triggers should be placed in. It is 
  designed to be plugged in to some other framework that handles looping over events in whatever
  samples of interest; a HLTDatasets object just collects the information it is fed within the
  event loop, and upon request outputs diagnostics using whatever it has been given so far.

  The primary datasets are defined in an input text file, which is just a collection of blocks of
  the format:
  @verbatim
    PrimaryDatasetA:
      TriggerPath1
      TriggerPath2
      ...
  @endverbatim
  Whitespace is irrelevant, but do remember that a colon (:) is used to indicate the start of the
  next primary dataset definition block. To-end-of-line comments beginning with # are allowed.

  The first step after creating a HLTDatasets object, is to register all the samples that will be
  run over via calls like:
  @code
    HLTDatasets   hltDatasets(triggerNames, "Datasets_1e31.list");
    hltDatasets.addSample("MinBias", RATE_SAMPLE);
    hltDatasets.addSample("TTbar"  , PHYSICS_SAMPLE);
  @endcode
  and so forth. Then inside the event loop for a particular sample (which you have to code for
  yourself), the SampleDiagnostics for that sample should be updated after all the trigger bits
  for that event are available:
  @code
    for (Long64_t eventIndex = 0; eventIndex < numEventsInSample; ++eventIndex) {
      ...
      hltDatasets[sampleIndex].fill(triggerBit);
      ...
    } // end loop over events in tree
  @endcode
  At the end of all event loops, the event counts should be converted into rates:
  @code
    Double_t    collisionRate = nFilledBunches / maxFilledBunches / bunchCrossingTime;
    for (Int_t sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
      Double_t  mu            = bunchCrossingTime * crossSection * instantaneousLuminosity * maxFilledBunches / nFilledBunches;
      hltDatasets[sampleIndex].computeRate(collisionRate, mu);
    } // end loop over events in tree
  @endcode
  Finally, the diagnostics can be output into the path of choice (default current directory) via:
  @code
    hltDatasets.report("outputPath/descriptivePrefix_");
  @endcode


  One ROOT file and one PDF file are made, containing the following diagnostics:
    - New-trigger vs. primary dataset correlation plot: every cell in this 2D plot is the overlap 
      fraction [RateIn(X AND Y) / RateIn(X)], where X and Y are either new triggers or 
      primary datasets (PD), as seen on the label on the x and y axes respectively. The total overhead
      [RateIn(PD1) + RateIn(PD2) + ... + RateIn(PDN)] / RateIn(any PD)]
      is shown in the title. This overhead does not include the new triggers, just the existing
      primary datasets.
    - For each new trigger, a table showing what the impact on the rate of each primary dataset (PD) 
      would be, if the trigger were to be added to that particular dataset. 

  Because it is often of interest to understand what a trigger is doing for a particular sample, the
  above diagnostics are actually repeated for each of the following collections:
    - Sum of all samples.
    - Sum of all samples designated as RATE_SAMPLE.
    - Sum of all samples designated as PHYSICS_SAMPLE.
    - Individually for each sample in the set designated as PHYSICS_SAMPLE.
*/
class HLTDatasets : protected std::vector<SampleDiagnostics>
{
  //.. Data ...................................................................
protected:
  SampleDiagnostics     datasetsConfig;     ///< Template containing the primary dataset and new trigger specifications, as loaded from the definitions file at construction time.
  TString               scenarioName;       ///< Name (minus extension) of the user-provided datasetDefinitionFile, to be used 
  TStopwatch            timer;              ///< For bench-marking the execution, if you care.

public:
  using std::vector<SampleDiagnostics>::operator[];
  using std::vector<SampleDiagnostics>::at;
  using std::vector<SampleDiagnostics>::size;

  //.. Functions ..............................................................
  /**
    Creates a HLTDatasets object for a particular set of primary dataset definitions and triggers.
    Don't forget to register all the samples to be run over via addSample() after this.
  */
	HLTDatasets ( const std::vector<TString>&   triggerNames                      ///< The list of trigger names. All triggers not contained in any primary dataset (as specified in datasetDefinitionFile) will be considered to be a "new" trigger. The order of triggers in this list is the order in which they appear in the diagnostics.
              , const Char_t*                 datasetDefinitionFile             ///< The path of the file that contains the primary dataset definitions. All triggers @e not within any primary dataset, will be considered as a "new" trigger for which correlations and rates will be investigated. If no file is specified, assumes all triggers are new -- this is a way to get the trigger-vs-trigger correlations.
              , Bool_t                        preferEmulatedTriggers  = kTRUE   ///< If set, always picks the emulation of the trigger when both emulated and original are available. A trigger is considered to be emulated if its name starts with emulationPrefix.
              , TString                       emulationPrefix         = "Open"  ///< The prefix by which to determine whether or not a trigger is an emulated version of a path with the same name except for the prefix. If a trigger has both an emulated and original version, only one of the choices will be used for this analysis.
              );
  ~HLTDatasets();

  /**
    Registers a sample in the list that will be processed in future event loops. The separation into
    samples is for book-keeping purposes.
  */
  void addSample( const Char_t     sampleName[]  = ""            ///< Name of the sample, for display purposes.
                , SampleCategory   typeOfSample  = RATE_SAMPLE   ///< The type of sample (see the SampleCategory documentation), for presentation purposes.
                );

  /**
    Outputs the diagnostics as discussed in the HLTDatasets introduction:
      - New-trigger vs. primary dataset correlation plot

    One ROOT file 
    @verbatim
      scenario_correlations.root 
    @endverbatim
    is created containing all the correlation plots (which are labeled according to sample). 
    @c scenario is the name (minus extension) of the user-specified 
    datasetDefinitionFile.
  */
  void  write ( const Char_t*   outputPrefix      = 0           ///< Optional prefix for the output files. You can use this to put the output in your directory of choice (don't forget the trailing slash). Directories are automatically created as necessary.
              , Option_t*       writeOptions      = "RECREATE"  ///< Options for the ROOT file creation (TFile::TFile()).
              ) const;

  /**
    Outputs the diagnostics as discussed in the HLTDatasets introduction:
      - For each new trigger, a table showing what the impact on the rate of each primary dataset (PD) 
        would be, if the trigger were to be added to that particular dataset

    One PDF file 
    @verbatim
      scenario_newtriggers.pdf
    @endverbatim
    is created containing the table of contributions of each new trigger. The tables are 
    further broken down into separate tables for all the samples designated as PHYSICS_SAMPLE,
    and also the @c PHYSICS_SAMPLE and @c RATE_SAMPLE compilations of samples.
  */
  void report ( const Char_t*   luminosity        = 0           ///< Instantaneous luminosity, for display purposes if provided.
              , const Char_t*   outputPrefix      = 0           ///< Optional prefix for the output files. You can use this to put the output in your directory of choice (don't forget the trailing slash). Directories are automatically created as necessary.
              , const Int_t     significantDigits = 3           ///< Number of significant digits to report percentages in.
              ) const;

protected:
  /**
    Creates "all", "rate", and "physics" compilations of samples (in this order).
    Appends all existing samples after those three. Returns the total number of 
    diagnostics.
  */
  UInt_t  compileSamples( std::vector<SampleDiagnostics>&   compiledDiagnostics   ///< The output compilations.
                        ) const;
};




#endif //__HLTDATASETS_H__
