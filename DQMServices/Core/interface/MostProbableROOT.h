// QTest that should test Most Probable value for some Expected number

// Author : Samvel Khalatian ( samvel at fnal dot gov )
// Created: 04/26/07

#ifndef MOST_PROBABLE_ROOT_H
#define MOST_PROBABLE_ROOT_H

#include <string>

#include "DQMServices/Core/interface/QualTestBase.h"

// Forward declarations
class TH1F;

namespace edm {
  namespace qtests {
    namespace fits {
      // Convert Significance into Probability value.
      double erfc( const double &rdX);
    }
  }
}

/** 
* @brief 
*   Base for all MostProbables Children classes. Thus each child 
*   implementation will concentrate on fit itself.
*/
class MostProbableBaseROOT: public SimpleTest<TH1F> {
  public:
    MostProbableBaseROOT();

    virtual ~MostProbableBaseROOT() {}

    /** 
    * @brief 
    *   Actual Run Test method. Should return: [0, 1] or <0 for failure.
    *   [Note: See SimpleTest<class T> template for details]
    * 
    * @param poPLOT  Plot for Which QTest to be run
    * 
    * @return 
    *   -1      On Error
    *   [0, 1]  Measurement of how Fit value is close to Expected one
    */
    virtual float runTest( const TH1F *const poPLOT);

    // Set/Get local variables methods
    inline void   setMostProbable( const double &rdMP) { dMostProbable_ = rdMP;}
    inline double getMostProbable() const              { return dMostProbable_;}

    inline void   setSigma( const double &rdSIGMA) { dSigma_ = rdSIGMA; }
    inline double getSigma() const                 { return dSigma_; }

    inline void   setXMin( const double &rdMIN) { dXMin_ = rdMIN; }
    inline double getXMin() const               { return dXMin_;  }

    inline void   setXMax( const double &rdMAX) { dXMax_ = rdMAX; }
    inline double getXMax() const               { return dXMax_;  }

  protected:
    /** 
    * @brief 
    *   Each Child should implement fit method which responsibility is to 
    *   perform actual fit and compare mean value with some additional
    *   Cuts if any needed. The reason this task is put into separate method
    *   is that a priory it is unknown what distribution QTest is dealing with.
    *   It might be simple Gauss, Landau or something more sophisticated.
    *   Each Plot needs special treatment (fitting) and extraction of 
    *   parameters. Children know about that but not Parent class.
    * 
    * @param poPLOT  Plot to be fitted
    * 
    * @return 
    *   -1     On Error
    *   [0,1]  Measurement of how close Fit Value is to Expected one
    */
    virtual float fit( TH1F *const poPLOT) = 0;

    /** 
    * @brief 
    *   Child should check test if it is valid and return corresponding value
    *   Next common tests are performed here:
    *     1. min < max
    *     2. MostProbable is in (min, max)
    *     3. Sigma > 0
    * 
    * @return 
    *   True   Invalid QTest
    *   False  Otherwise
    */
    virtual bool isInvalid();

    /** 
    * @brief 
    *   General function that compares MostProbable value gotten from Fit and
    *   Expected one.
    * 
    * @param rdMP_FIT     MostProbable value gotten from Fit
    * @param rdSIGMA_FIT  Sigma value gotten from Fit
    * 
    * @return 
    *   Probability of found Value that measures how close is gotten one to 
    *   expected
    */
    double compareMostProbables( const double &rdMP_FIT, 
                                 const double &rdSIGMA_FIT) const;

  private:
    // Most common Fit values
    double dMostProbable_;
    double dSigma_;
    double dXMin_;
    double dXMax_;
};

// --[ Fit: Landau ]-----------------------------------------------------------
/** 
* @brief 
*   MostProbable QTest for Landau distributions
*/
class MostProbableLandauROOT: public MostProbableBaseROOT {
  public:
    MostProbableLandauROOT();
    virtual ~MostProbableLandauROOT() {}

    /** 
    * @brief 
    *   Each QTest should define given method in order to identify itself.
    * 
    * @return 
    *   QTest identification name
    */
    static inline std::string getAlgoName() { return "MostProbableLandau"; }

    // Set/Get local variables methods
    inline void setNormalization( const double &rdNORMALIZATION) {
      dNormalization_ = rdNORMALIZATION;
    }
    inline double getNormalization() const { return dNormalization_; }

  protected:
    /** 
    * @brief
    *   Perform Actual Fit
    * 
    * @param poPLOT  Plot to be fitted
    * 
    * @return 
    *   -1     On Error
    *   [0,1]  Measurement of how close Fit Value is to Expected one
    */
    virtual float fit( TH1F *const poPLOT);

  private:
    double dNormalization_;
};

// --[ Fit: Gauss ]------------------------------------------------------------

// --[ Fit: Landau + Gauss ]---------------------------------------------------

#endif // MOST_PROBABLE_ROOT_H
