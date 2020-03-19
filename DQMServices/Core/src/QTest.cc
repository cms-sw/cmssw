#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/src/DQMError.h"
#include "Math/ProbFuncMathCore.h"
#include "TMath.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;

const float QCriterion::ERROR_PROB_THRESHOLD = 0.50;
const float QCriterion::WARNING_PROB_THRESHOLD = 0.90;

// initialize values
void QCriterion::init() {
  errorProb_ = ERROR_PROB_THRESHOLD;
  warningProb_ = WARNING_PROB_THRESHOLD;
  setAlgoName("NO_ALGORITHM");
  status_ = dqm::qstatus::DID_NOT_RUN;
  message_ = "NO_MESSAGE";
  verbose_ = 0;  // 0 = silent, 1 = algorithmic failures, 2 = info
}

float QCriterion::runTest(const MonitorElement* /* me */) {
  raiseDQMError("QCriterion", "virtual runTest method called");
  return 0.;
}
//===================================================//
//================ QUALITY TESTS ====================//
//==================================================//

//----------------------------------------------------//
//--------------- ContentsXRange ---------------------//
//----------------------------------------------------//
float ContentsXRange::runTest(const MonitorElement* me) {
  badChannels_.clear();

  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";
  // -- TH1F
  if (me->kind() == MonitorElement::Kind::TH1F) {
    h = me->getTH1F();
  }
  // -- TH1S
  else if (me->kind() == MonitorElement::Kind::TH1S) {
    h = me->getTH1S();
  }
  // -- TH1D
  else if (me->kind() == MonitorElement::Kind::TH1D) {
    h = me->getTH1D();
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:ContentsXRange"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D, exiting\n";
    return -1;
  }

  if (!rangeInitialized_) {
    if (h->GetXaxis())
      setAllowedXRange(h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
    else
      return -1;
  }
  int ncx = h->GetXaxis()->GetNbins();
  // use underflow bin
  int first = 0;  // 1
  // use overflow bin
  int last = ncx + 1;  // ncx
  // all entries
  double sum = 0;
  // entries outside X-range
  double fail = 0;
  int bin;
  for (bin = first; bin <= last; ++bin) {
    double contents = h->GetBinContent(bin);
    double x = h->GetBinCenter(bin);
    sum += contents;
    if (x < xmin_ || x > xmax_)
      fail += contents;
  }

  if (sum == 0)
    return 1;
  // return fraction of entries within allowed X-range
  return (sum - fail) / sum;
}

//-----------------------------------------------------//
//--------------- ContentsYRange ---------------------//
//----------------------------------------------------//
float ContentsYRange::runTest(const MonitorElement* me) {
  badChannels_.clear();

  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";

  if (me->kind() == MonitorElement::Kind::TH1F) {
    h = me->getTH1F();  //access Test histo
  } else if (me->kind() == MonitorElement::Kind::TH1S) {
    h = me->getTH1S();  //access Test histo
  } else if (me->kind() == MonitorElement::Kind::TH1D) {
    h = me->getTH1D();  //access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:ContentsYRange"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D, exiting\n";
    return -1;
  }

  if (!rangeInitialized_ || !h->GetXaxis())
    return 1;  // all bins are accepted if no initialization
  int ncx = h->GetXaxis()->GetNbins();
  // do NOT use underflow bin
  int first = 1;
  // do NOT use overflow bin
  int last = ncx;
  // bins outside Y-range
  int fail = 0;
  int bin;

  if (useEmptyBins_)  ///Standard test !
  {
    for (bin = first; bin <= last; ++bin) {
      double contents = h->GetBinContent(bin);
      bool failure = false;
      failure = (contents < ymin_ || contents > ymax_);  // allowed y-range: [ymin_, ymax_]
      if (failure) {
        DQMChannel chan(bin, 0, 0, contents, h->GetBinError(bin));
        badChannels_.push_back(chan);
        ++fail;
      }
    }
    // return fraction of bins that passed test
    return 1. * (ncx - fail) / ncx;
  } else  ///AS quality test !!!
  {
    for (bin = first; bin <= last; ++bin) {
      double contents = h->GetBinContent(bin);
      bool failure = false;
      if (contents)
        failure = (contents < ymin_ || contents > ymax_);  // allowed y-range: [ymin_, ymax_]
      if (failure)
        ++fail;
    }
    // return fraction of bins that passed test
    return 1. * (ncx - fail) / ncx;
  }  ///end of AS quality tests
}

//-----------------------------------------------------//
//------------------ DeadChannel ---------------------//
//----------------------------------------------------//
float DeadChannel::runTest(const MonitorElement* me) {
  badChannels_.clear();
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h1 = nullptr;
  TH2* h2 = nullptr;  //initialize histogram pointers

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";
  //TH1F
  if (me->kind() == MonitorElement::Kind::TH1F) {
    h1 = me->getTH1F();  //access Test histo
  }
  //TH1S
  else if (me->kind() == MonitorElement::Kind::TH1S) {
    h1 = me->getTH1S();  //access Test histo
  }
  //TH1D
  else if (me->kind() == MonitorElement::Kind::TH1D) {
    h1 = me->getTH1D();  //access Test histo
  }
  //-- TH2F
  else if (me->kind() == MonitorElement::Kind::TH2F) {
    h2 = me->getTH2F();  // access Test histo
  }
  //-- TH2S
  else if (me->kind() == MonitorElement::Kind::TH2S) {
    h2 = me->getTH2S();  // access Test histo
  }
  //-- TH2D
  else if (me->kind() == MonitorElement::Kind::TH2D) {
    h2 = me->getTH2D();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:DeadChannel"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D/TH2F/TH2S/TH2D, exiting\n";
    return -1;
  }

  int fail = 0;  // number of failed channels

  //--------- do the quality test for 1D histo ---------------//
  if (h1 != nullptr) {
    if (!rangeInitialized_ || !h1->GetXaxis())
      return 1;  // all bins are accepted if no initialization
    int ncx = h1->GetXaxis()->GetNbins();
    int first = 1;
    int last = ncx;
    int bin;

    /// loop over all channels
    for (bin = first; bin <= last; ++bin) {
      double contents = h1->GetBinContent(bin);
      bool failure = false;
      failure = contents <= ymin_;  // dead channel: equal to or less than ymin_
      if (failure) {
        DQMChannel chan(bin, 0, 0, contents, h1->GetBinError(bin));
        badChannels_.push_back(chan);
        ++fail;
      }
    }
    //return fraction of alive channels
    return 1. * (ncx - fail) / ncx;
  }
  //----------------------------------------------------------//

  //--------- do the quality test for 2D -------------------//
  else if (h2 != nullptr) {
    int ncx = h2->GetXaxis()->GetNbins();  // get X bins
    int ncy = h2->GetYaxis()->GetNbins();  // get Y bins

    /// loop over all bins
    for (int cx = 1; cx <= ncx; ++cx) {
      for (int cy = 1; cy <= ncy; ++cy) {
        double contents = h2->GetBinContent(h2->GetBin(cx, cy));
        bool failure = false;
        failure = contents <= ymin_;  // dead channel: equal to or less than ymin_
        if (failure) {
          DQMChannel chan(cx, cy, 0, contents, h2->GetBinError(h2->GetBin(cx, cy)));
          badChannels_.push_back(chan);
          ++fail;
        }
      }
    }
    //return fraction of alive channels
    return 1. * (ncx * ncy - fail) / (ncx * ncy);
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:DeadChannel"
                << " TH1/TH2F are NULL, exiting\n";
    return -1;
  }
}

//-----------------------------------------------------//
//----------------  NoisyChannel ---------------------//
//----------------------------------------------------//
// run the test (result: fraction of channels not appearing noisy or "hot")
// [0, 1] or <0 for failure
float NoisyChannel::runTest(const MonitorElement* me) {
  badChannels_.clear();
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;   //initialize histogram pointer
  TH2* h2 = nullptr;  //initialize histogram pointer

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";

  int nbins = 0;
  int nbinsX = 0, nbinsY = 0;
  //-- TH1F
  if (me->kind() == MonitorElement::Kind::TH1F) {
    nbins = me->getTH1F()->GetXaxis()->GetNbins();
    h = me->getTH1F();  // access Test histo
  }
  //-- TH1S
  else if (me->kind() == MonitorElement::Kind::TH1S) {
    nbins = me->getTH1S()->GetXaxis()->GetNbins();
    h = me->getTH1S();  // access Test histo
  }
  //-- TH1D
  else if (me->kind() == MonitorElement::Kind::TH1D) {
    nbins = me->getTH1D()->GetXaxis()->GetNbins();
    h = me->getTH1D();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2F) {
    nbinsX = me->getTH2F()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2F()->GetYaxis()->GetNbins();
    h2 = me->getTH2F();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2S) {
    nbinsX = me->getTH2S()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2S()->GetYaxis()->GetNbins();
    h2 = me->getTH2S();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2D) {
    nbinsX = me->getTH2F()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2F()->GetYaxis()->GetNbins();
    h2 = me->getTH2D();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:NoisyChannel"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D or TH2F/TH2S/TH2D, exiting\n";
    return -1;
  }

  //--  QUALITY TEST itself

  // do NOT use underflow bin
  int first = 1;
  // do NOT use overflow bin
  int last = nbins;
  int lastX = nbinsX, lastY = nbinsY;
  // bins outside Y-range
  int fail = 0;
  int bin;
  int binX, binY;
  if (h != nullptr) {
    if (!rangeInitialized_ || !h->GetXaxis()) {
      return 1;  // all channels are accepted if tolerance has not been set
    }
    for (bin = first; bin <= last; ++bin) {
      double contents = h->GetBinContent(bin);
      double average = getAverage(bin, h);
      bool failure = false;
      if (average != 0)
        failure = (((contents - average) / std::abs(average)) > tolerance_);

      if (failure) {
        ++fail;
        DQMChannel chan(bin, 0, 0, contents, h->GetBinError(bin));
        badChannels_.push_back(chan);
      }
    }

    // return fraction of bins that passed test
    return 1. * (nbins - fail) / nbins;
  } else if (h2 != nullptr) {
    for (binY = first; binY <= lastY; ++binY) {
      for (binX = first; binX <= lastX; ++binX) {
        double contents = h2->GetBinContent(binX, binY);
        double average = getAverage2D(binX, binY, h2);
        bool failure = false;
        if (average != 0)
          failure = (((contents - average) / std::abs(average)) > tolerance_);
        if (failure) {
          ++fail;
          DQMChannel chan(binX, 0, 0, contents, h2->GetBinError(binX));
          badChannels_.push_back(chan);
        }
      }  //end x loop
    }    //end y loop
    // return fraction of bins that passed test
    return 1. * ((nbinsX * nbinsY) - fail) / (nbinsX * nbinsY);
  }  //end nullptr conditional
  else {
    if (verbose_ > 0)
      std::cout << "QTest:NoisyChannel"
                << " TH1/TH2F are NULL, exiting\n";
    return -1;
  }
}

// get average for bin under consideration
// (see description of method setNumNeighbors)
double NoisyChannel::getAverage(int bin, const TH1* h) const {
  int first = 1;                        // Do NOT use underflow bin
  int ncx = h->GetXaxis()->GetNbins();  // Do NOT use overflow bin
  double sum = 0;
  int bin_start, bin_end;
  int add_right = 0;
  int add_left = 0;

  bin_start = bin - numNeighbors_;  // First bin in integral
  bin_end = bin + numNeighbors_;    // Last bin in integral

  if (bin_start < first) {          // If neighbors take you outside of histogram range shift integral right
    add_right = first - bin_start;  // How much to shift remembering we are not using underflow
    bin_start = first;              // Remember to reset the starting bin
    bin_end += add_right;
    if (bin_end > ncx)
      bin_end = ncx;  // If the test would be larger than histogram just sum histogram without overflow
  }

  if (bin_end > ncx) {  // Now we make sure doesn't run off right edge of histogram
    add_left = bin_end - ncx;
    bin_end = ncx;
    bin_start -= add_left;
    if (bin_start < first)
      bin_start = first;  // If the test would be larger than histogram just sum histogram without underflow
  }

  sum += h->Integral(bin_start, bin_end);
  sum -= h->GetBinContent(bin);

  int dimension = 2 * numNeighbors_ + 1;
  if (dimension > h->GetNbinsX())
    dimension = h->GetNbinsX();

  return sum / (dimension - 1);
}

double NoisyChannel::getAverage2D(int binX, int binY, const TH2* h2) const {
  /// Do NOT use underflow or overflow bins
  int firstX = 1;
  int firstY = 1;
  double sum = 0;
  int ncx = h2->GetXaxis()->GetNbins();
  int ncy = h2->GetYaxis()->GetNbins();

  int neighborsX, neighborsY;  // Convert unsigned input to int so we can use comparators
  neighborsX = numNeighbors_;
  neighborsY = numNeighbors_;
  int bin_startX, bin_endX;
  int add_rightX = 0;  // Start shifts at 0
  int add_leftX = 0;
  int bin_startY, bin_endY;
  int add_topY = 0;
  int add_downY = 0;

  bin_startX = binX - neighborsX;  // First bin in X
  bin_endX = binX + neighborsX;    // Last bin in X

  if (bin_startX < firstX) {           // If neighbors take you outside of histogram range shift integral right
    add_rightX = firstX - bin_startX;  // How much to shift remembering we are no using underflow
    bin_startX = firstX;               // Remember to reset the starting bin
    bin_endX += add_rightX;
    if (bin_endX > ncx)
      bin_endX = ncx;
  }

  if (bin_endX > ncx) {  // Now we make sure doesn't run off right edge of histogram
    add_leftX = bin_endX - ncx;
    bin_endX = ncx;
    bin_startX -= add_leftX;
    if (bin_startX < firstX)
      bin_startX = firstX;  // If the test would be larger than histogram just sum histogram without underflow
  }

  bin_startY = binY - neighborsY;  // First bin in Y
  bin_endY = binY + neighborsY;    // Last bin in Y

  if (bin_startY < firstY) {         // If neighbors take you outside of histogram range shift integral up
    add_topY = firstY - bin_startY;  // How much to shift remembering we are no using underflow
    bin_startY = firstY;             // Remember to reset the starting bin
    bin_endY += add_topY;
    if (bin_endY > ncy)
      bin_endY = ncy;
  }

  if (bin_endY > ncy) {  // Now we make sure doesn't run off top edge of histogram
    add_downY = bin_endY - ncy;
    bin_endY = ncy;
    bin_startY -= add_downY;
    if (bin_startY < firstY)
      bin_startY = firstY;  // If the test would be larger than histogram just sum histogram without underflow
  }

  sum += h2->Integral(bin_startX, bin_endX, bin_startY, bin_endY);
  sum -= h2->GetBinContent(binX, binY);

  int dimensionX = 2 * neighborsX + 1;
  int dimensionY = 2 * neighborsY + 1;

  if (dimensionX > h2->GetNbinsX())
    dimensionX = h2->GetNbinsX();
  if (dimensionY > h2->GetNbinsY())
    dimensionY = h2->GetNbinsY();

  return sum / (dimensionX * dimensionY - 1);  // Average is sum over the # of bins used

}  // End getAverage2D

//-----------------------------------------------------//
//-----Content Sigma (Emma Yeager and Chad Freer)------//
//----------------------------------------------------//
// run the test (result: fraction of channels with sigma that is not noisy or hot)

float ContentSigma::runTest(const MonitorElement* me) {
  badChannels_.clear();
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;  //initialize histogram pointer

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";

  unsigned nbinsX;
  unsigned nbinsY;

  //-- TH1F
  if (me->kind() == MonitorElement::Kind::TH1F) {
    nbinsX = me->getTH1F()->GetXaxis()->GetNbins();
    nbinsY = me->getTH1F()->GetYaxis()->GetNbins();
    h = me->getTH1F();  // access Test histo
  }
  //-- TH1S
  else if (me->kind() == MonitorElement::Kind::TH1S) {
    nbinsX = me->getTH1S()->GetXaxis()->GetNbins();
    nbinsY = me->getTH1S()->GetYaxis()->GetNbins();
    h = me->getTH1S();  // access Test histo
  }
  //-- TH1D
  else if (me->kind() == MonitorElement::Kind::TH1D) {
    nbinsX = me->getTH1D()->GetXaxis()->GetNbins();
    nbinsY = me->getTH1D()->GetYaxis()->GetNbins();
    h = me->getTH1D();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2F) {
    nbinsX = me->getTH2F()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2F()->GetYaxis()->GetNbins();
    h = me->getTH2F();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2S) {
    nbinsX = me->getTH2S()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2S()->GetYaxis()->GetNbins();
    h = me->getTH2S();  // access Test histo
  }
  //-- TH2
  else if (me->kind() == MonitorElement::Kind::TH2D) {
    nbinsX = me->getTH2D()->GetXaxis()->GetNbins();
    nbinsY = me->getTH2D()->GetYaxis()->GetNbins();
    h = me->getTH2D();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:ContentSigma"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D or TH2F/TH2S/TH2D, exiting\n";
    return -1;
  }

  //--  QUALITY TEST itself

  if (!rangeInitialized_ || !h->GetXaxis())
    return 1;  // all channels are accepted if tolerance has not been set

  int fail = 0;       // initialize bin failure count
  unsigned xMin = 1;  //initialize minimums and maximums with expected values
  unsigned yMin = 1;
  unsigned xMax = nbinsX;
  unsigned yMax = nbinsY;
  unsigned XBlocks = numXblocks_;  //Initialize xml inputs blocks and neighbors
  unsigned YBlocks = numYblocks_;
  unsigned neighborsX = numNeighborsX_;
  unsigned neighborsY = numNeighborsY_;
  unsigned Xbinnum = 1;
  unsigned Ybinnum = 1;
  unsigned XWidth = 0;
  unsigned YWidth = 0;

  if (neighborsX == 999) {
    neighborsX = 0;
  }
  if (neighborsY == 999) {
    neighborsY = 0;
  }

  //give users option for automatic mininum and maximum selection by inputting 0 to any of the parameters
  // check that user's parameters are completely in agreement with histogram
  // for instance, if inputted xMax is out of range xMin will automatically be ignored
  if (xMin_ != 0 && xMax_ != 0) {
    if ((xMax_ <= nbinsX) && (xMin_ <= xMax_)) {  // rescale area of histogram being analyzed
      nbinsX = xMax_ - xMin_ + 1;
      xMax = xMax_;  // do NOT use overflow bin
      xMin = xMin_;  // do NOT use underflow bin
    }
  }
  //give users option for automatic mininum and maximum selection by inputting 0 to any of the parameters
  if (yMin_ != 0 && yMax_ != 0) {
    if ((yMax_ <= nbinsY) && (yMin_ <= yMax_)) {
      nbinsY = yMax_ - yMin_ + 1;
      yMax = yMax_;
      yMin = yMin_;
    }
  }

  if (neighborsX * 2 >= nbinsX) {  //make sure neighbor check does not overlap with bin under consideration
    if (nbinsX % 2 == 0) {
      neighborsX = nbinsX / 2 - 1;  //set neighbors for no overlap
    } else {
      neighborsX = (nbinsX - 1) / 2;
    }
  }

  if (neighborsY * 2 >= nbinsY) {
    if (nbinsY % 2 == 0) {
      neighborsY = nbinsY / 2 - 1;
    } else {
      neighborsY = (nbinsY - 1) / 2;
    }
  }

  if (XBlocks == 999) {  //Setting 999 prevents blocks and does quality tests by bins only
    XBlocks = nbinsX;
  }
  if (YBlocks == 999) {
    YBlocks = nbinsY;
  }

  Xbinnum = nbinsX / XBlocks;
  Ybinnum = nbinsY / YBlocks;
  for (unsigned groupx = 0; groupx < XBlocks; ++groupx) {  //Test over all the blocks
    for (unsigned groupy = 0; groupy < YBlocks; ++groupy) {
      double blocksum = 0;
      for (unsigned binx = 0; binx < Xbinnum; ++binx) {  //Sum the contents of the block in question
        for (unsigned biny = 0; biny < Ybinnum; ++biny) {
          if (groupx * Xbinnum + xMin + binx <= xMax && groupy * Ybinnum + yMin + biny <= yMax) {
            blocksum += abs(h->GetBinContent(groupx * Xbinnum + xMin + binx, groupy * Ybinnum + yMin + biny));
          }
        }
      }

      double sum = getNeighborSum(groupx, groupy, XBlocks, YBlocks, neighborsX, neighborsY, h);
      sum -= blocksum;  //remove center block to test

      if (neighborsX > groupx) {  //Find correct average at the edges
        XWidth = neighborsX + groupx + 1;
      } else if (neighborsX > (XBlocks - (groupx + 1))) {
        XWidth = (XBlocks - groupx) + neighborsX;
      } else {
        XWidth = 2 * neighborsX + 1;
      }
      if (neighborsY > groupy) {
        YWidth = neighborsY + groupy + 1;
      } else if (neighborsY > (YBlocks - (groupy + 1))) {
        YWidth = (YBlocks - groupy) + neighborsY;
      } else {
        YWidth = 2 * neighborsY + 1;
      }

      double average = sum / (XWidth * YWidth - 1);
      double sigma = getNeighborSigma(average, groupx, groupy, XBlocks, YBlocks, neighborsX, neighborsY, h);
      //get rid of block being tested just like we did with the average
      sigma -= (average - blocksum) * (average - blocksum);
      double sigma_2 = sqrt(sigma) / sqrt(XWidth * YWidth - 2);  //N-1 where N=XWidth*YWidth - 1
      double sigma_real = sigma_2 / (XWidth * YWidth - 1);
      //double avg_uncrt = average*sqrt(sum)/sum;//Obsolete now(Chad Freer)
      double avg_uncrt = 3 * sigma_real;

      double probNoisy = ROOT::Math::poisson_cdf_c(blocksum - 1, average + avg_uncrt);
      double probDead = ROOT::Math::poisson_cdf(blocksum, average - avg_uncrt);
      double thresholdNoisy = ROOT::Math::normal_cdf_c(toleranceNoisy_);
      double thresholdDead = ROOT::Math::normal_cdf(-1 * toleranceDead_);

      int failureNoisy = 0;
      int failureDead = 0;

      if (average != 0) {
        if (noisy_ && dead_) {
          if (blocksum > average) {
            failureNoisy = probNoisy < thresholdNoisy;
          } else {
            failureDead = probDead < thresholdDead;
          }
        } else if (noisy_) {
          if (blocksum > average) {
            failureNoisy = probNoisy < thresholdNoisy;
          }
        } else if (dead_) {
          if (blocksum < average) {
            failureDead = probDead < thresholdDead;
          }
        } else {
          std::cout << "No test type selected!\n";
        }
        //Following lines useful for debugging using verbose (Chad Freer)
        //string histName = h->GetName();
        //if (histName == "emtfTrackBX") {
        //   std::printf("Chad says: %i XBlocks, %i XBlocks, %f Blocksum, %f Average", XBlocks,YBlocks,blocksum,average);}
      }

      if (failureNoisy || failureDead) {
        ++fail;
        //DQMChannel chan(groupx*Xbinnum+xMin+binx, 0, 0, blocksum, h->GetBinError(groupx*Xbinnum+xMin+binx));
        //badChannels_.push_back(chan);
      }
    }
  }
  return 1. * ((XBlocks * YBlocks) - fail) / (XBlocks * YBlocks);
}

//Gets the sum of the bins surrounding the block to be tested (Chad Freer)
double ContentSigma::getNeighborSum(unsigned groupx,
                                    unsigned groupy,
                                    unsigned Xblocks,
                                    unsigned Yblocks,
                                    unsigned neighborsX,
                                    unsigned neighborsY,
                                    const TH1* h) const {
  double sum = 0;
  unsigned nbinsX = h->GetXaxis()->GetNbins();
  unsigned nbinsY = h->GetYaxis()->GetNbins();
  unsigned xMin = 1;
  unsigned yMin = 1;
  unsigned xMax = nbinsX;
  unsigned yMax = nbinsY;
  unsigned Xbinnum = 1;
  unsigned Ybinnum = 1;

  //give users option for automatic mininum and maximum selection by inputting 0 to any of the parameters
  // check that user's parameters are completely in agreement with histogram
  // for instance, if inputted xMax is out of range xMin will automatically be ignored
  if (xMin_ != 0 && xMax_ != 0) {
    if ((xMax_ <= nbinsX) && (xMin_ <= xMax_)) {
      nbinsX = xMax_ - xMin_ + 1;
      xMax = xMax_;  // do NOT use overflow bin
      xMin = xMin_;  // do NOT use underflow bin
    }
  }
  if (yMin_ != 0 && yMax_ != 0) {
    if ((yMax_ <= nbinsY) && (yMin_ <= yMax_)) {
      nbinsY = yMax_ - yMin_ + 1;
      yMax = yMax_;
      yMin = yMin_;
    }
  }

  if (Xblocks == 999) {  //Check to see if blocks should be ignored
    Xblocks = nbinsX;
  }
  if (Yblocks == 999) {
    Yblocks = nbinsY;
  }

  Xbinnum = nbinsX / Xblocks;
  Ybinnum = nbinsY / Yblocks;

  unsigned xLow, xHi, yLow, yHi;  //Define the neighbor blocks edges to be summed
  if (groupx > neighborsX) {
    xLow = (groupx - neighborsX) * Xbinnum + xMin;
    if (xLow < xMin) {
      xLow = xMin;  //If the neigbor block would go outside the histogram edge, set it the edge
    }
  } else {
    xLow = xMin;
  }
  xHi = (groupx + 1 + neighborsX) * Xbinnum + xMin;
  if (xHi > xMax) {
    xHi = xMax;
  }
  if (groupy > neighborsY) {
    yLow = (groupy - neighborsY) * Ybinnum + yMin;
    if (yLow < yMin) {
      yLow = yMin;
    }
  } else {
    yLow = yMin;
  }
  yHi = (groupy + 1 + neighborsY) * Ybinnum + yMin;
  if (yHi > yMax) {
    yHi = yMax;
  }

  for (unsigned xbin = xLow; xbin <= xHi; ++xbin) {  //now sum over all the bins
    for (unsigned ybin = yLow; ybin <= yHi; ++ybin) {
      sum += h->GetBinContent(xbin, ybin);
    }
  }
  return sum;
}

//Similar to algorithm  above but returns a version of standard deviation. Additional operations to return real standard deviation used above (Chad Freer)
double ContentSigma::getNeighborSigma(double average,
                                      unsigned groupx,
                                      unsigned groupy,
                                      unsigned Xblocks,
                                      unsigned Yblocks,
                                      unsigned neighborsX,
                                      unsigned neighborsY,
                                      const TH1* h) const {
  double sigma = 0;
  unsigned nbinsX = h->GetXaxis()->GetNbins();
  unsigned nbinsY = h->GetYaxis()->GetNbins();
  unsigned xMin = 1;
  unsigned yMin = 1;
  unsigned xMax = nbinsX;
  unsigned yMax = nbinsY;
  unsigned Xbinnum = 1;
  unsigned Ybinnum = 1;
  double block_sum;

  if (xMin_ != 0 && xMax_ != 0) {
    if ((xMax_ <= nbinsX) && (xMin_ <= xMax_)) {
      nbinsX = xMax_ - xMin_ + 1;
      xMax = xMax_;
      xMin = xMin_;
    }
  }
  if (yMin_ != 0 && yMax_ != 0) {
    if ((yMax_ <= nbinsY) && (yMin_ <= yMax_)) {
      nbinsY = yMax_ - yMin_ + 1;
      yMax = yMax_;
      yMin = yMin_;
    }
  }
  if (Xblocks == 999) {
    Xblocks = nbinsX;
  }
  if (Yblocks == 999) {
    Yblocks = nbinsY;
  }

  Xbinnum = nbinsX / Xblocks;
  Ybinnum = nbinsY / Yblocks;

  unsigned xLow, xHi, yLow, yHi;
  for (unsigned x_block_count = 0; x_block_count <= 2 * neighborsX; ++x_block_count) {
    for (unsigned y_block_count = 0; y_block_count <= 2 * neighborsY; ++y_block_count) {
      //Sum over blocks. Need to find standard deviation of average of blocksums. Set up low and hi values similar to sum but for blocks now.
      if (groupx + x_block_count > neighborsX) {
        xLow = (groupx + x_block_count - neighborsX) * Xbinnum + xMin;
        if (xLow < xMin) {
          xLow = xMin;
        }
      } else {
        xLow = xMin;
      }
      xHi = xLow + Xbinnum;
      if (xHi > xMax) {
        xHi = xMax;
      }
      if (groupy + y_block_count > neighborsY) {
        yLow = (groupy + y_block_count - neighborsY) * Ybinnum + yMin;
        if (yLow < yMin) {
          yLow = yMin;
        }
      } else {
        yLow = yMin;
      }
      yHi = yLow + Ybinnum;
      if (yHi > yMax) {
        yHi = yMax;
      }
      block_sum = 0;
      for (unsigned x_block_bin = xLow; x_block_bin <= xHi; ++x_block_bin) {
        for (unsigned y_block_bin = yLow; y_block_bin <= yHi; ++y_block_bin) {
          block_sum += h->GetBinContent(x_block_bin, y_block_bin);
        }
      }
      sigma += (average - block_sum) * (average - block_sum);  //will sqrt and divide by sqrt(N-1) outside of function
    }
  }
  return sigma;
}

//-----------------------------------------------------------//
//----------------  ContentsWithinExpected ---------------------//
//-----------------------------------------------------------//
// run the test (result: fraction of channels that passed test);
// [0, 1] or <0 for failure
float ContentsWithinExpected::runTest(const MonitorElement* me) {
  badChannels_.clear();
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;  //initialize histogram pointer

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";

  int ncx;
  int ncy;

  if (useEmptyBins_) {
    //-- TH2
    if (me->kind() == MonitorElement::Kind::TH2F) {
      ncx = me->getTH2F()->GetXaxis()->GetNbins();
      ncy = me->getTH2F()->GetYaxis()->GetNbins();
      h = me->getTH2F();  // access Test histo
    }
    //-- TH2S
    else if (me->kind() == MonitorElement::Kind::TH2S) {
      ncx = me->getTH2S()->GetXaxis()->GetNbins();
      ncy = me->getTH2S()->GetYaxis()->GetNbins();
      h = me->getTH2S();  // access Test histo
    }
    //-- TH2D
    else if (me->kind() == MonitorElement::Kind::TH2D) {
      ncx = me->getTH2D()->GetXaxis()->GetNbins();
      ncy = me->getTH2D()->GetYaxis()->GetNbins();
      h = me->getTH2D();  // access Test histo
    }
    //-- TProfile
    else if (me->kind() == MonitorElement::Kind::TPROFILE) {
      ncx = me->getTProfile()->GetXaxis()->GetNbins();
      ncy = 1;
      h = me->getTProfile();  // access Test histo
    }
    //-- TProfile2D
    else if (me->kind() == MonitorElement::Kind::TPROFILE2D) {
      ncx = me->getTProfile2D()->GetXaxis()->GetNbins();
      ncy = me->getTProfile2D()->GetYaxis()->GetNbins();
      h = me->getTProfile2D();  // access Test histo
    } else {
      if (verbose_ > 0)
        std::cout << "QTest:ContentsWithinExpected"
                  << " ME does not contain TH2F/TH2S/TH2D/TPROFILE/TPROFILE2D, exiting\n";
      return -1;
    }

    int nsum = 0;
    double sum = 0.0;
    double average = 0.0;

    if (checkMeanTolerance_) {  // calculate average value of all bin contents

      for (int cx = 1; cx <= ncx; ++cx) {
        for (int cy = 1; cy <= ncy; ++cy) {
          if (me->kind() == MonitorElement::Kind::TH2F) {
            sum += h->GetBinContent(h->GetBin(cx, cy));
            ++nsum;
          } else if (me->kind() == MonitorElement::Kind::TH2S) {
            sum += h->GetBinContent(h->GetBin(cx, cy));
            ++nsum;
          } else if (me->kind() == MonitorElement::Kind::TH2D) {
            sum += h->GetBinContent(h->GetBin(cx, cy));
            ++nsum;
          } else if (me->kind() == MonitorElement::Kind::TPROFILE) {
            if (me->getTProfile()->GetBinEntries(h->GetBin(cx)) >= minEntries_ / (ncx)) {
              sum += h->GetBinContent(h->GetBin(cx));
              ++nsum;
            }
          } else if (me->kind() == MonitorElement::Kind::TPROFILE2D) {
            if (me->getTProfile2D()->GetBinEntries(h->GetBin(cx, cy)) >= minEntries_ / (ncx * ncy)) {
              sum += h->GetBinContent(h->GetBin(cx, cy));
              ++nsum;
            }
          }
        }
      }

      if (nsum > 0)
        average = sum / nsum;

    }  // calculate average value of all bin contents

    int fail = 0;

    for (int cx = 1; cx <= ncx; ++cx) {
      for (int cy = 1; cy <= ncy; ++cy) {
        bool failMean = false;
        bool failRMS = false;
        bool failMeanTolerance = false;

        if (me->kind() == MonitorElement::Kind::TPROFILE &&
            me->getTProfile()->GetBinEntries(h->GetBin(cx)) < minEntries_ / (ncx))
          continue;

        if (me->kind() == MonitorElement::Kind::TPROFILE2D &&
            me->getTProfile2D()->GetBinEntries(h->GetBin(cx, cy)) < minEntries_ / (ncx * ncy))
          continue;

        if (checkMean_) {
          double mean = h->GetBinContent(h->GetBin(cx, cy));
          failMean = (mean < minMean_ || mean > maxMean_);
        }

        if (checkRMS_) {
          double rms = h->GetBinError(h->GetBin(cx, cy));
          failRMS = (rms < minRMS_ || rms > maxRMS_);
        }

        if (checkMeanTolerance_) {
          double mean = h->GetBinContent(h->GetBin(cx, cy));
          failMeanTolerance = (std::abs(mean - average) > toleranceMean_ * std::abs(average));
        }

        if (failMean || failRMS || failMeanTolerance) {
          if (me->kind() == MonitorElement::Kind::TH2F) {
            DQMChannel chan(cx, cy, 0, h->GetBinContent(h->GetBin(cx, cy)), h->GetBinError(h->GetBin(cx, cy)));
            badChannels_.push_back(chan);
          } else if (me->kind() == MonitorElement::Kind::TH2S) {
            DQMChannel chan(cx, cy, 0, h->GetBinContent(h->GetBin(cx, cy)), h->GetBinError(h->GetBin(cx, cy)));
            badChannels_.push_back(chan);
          } else if (me->kind() == MonitorElement::Kind::TH2D) {
            DQMChannel chan(cx, cy, 0, h->GetBinContent(h->GetBin(cx, cy)), h->GetBinError(h->GetBin(cx, cy)));
            badChannels_.push_back(chan);
          } else if (me->kind() == MonitorElement::Kind::TPROFILE) {
            DQMChannel chan(
                cx, cy, int(me->getTProfile()->GetBinEntries(h->GetBin(cx))), 0, h->GetBinError(h->GetBin(cx)));
            badChannels_.push_back(chan);
          } else if (me->kind() == MonitorElement::Kind::TPROFILE2D) {
            DQMChannel chan(cx,
                            cy,
                            int(me->getTProfile2D()->GetBinEntries(h->GetBin(cx, cy))),
                            h->GetBinContent(h->GetBin(cx, cy)),
                            h->GetBinError(h->GetBin(cx, cy)));
            badChannels_.push_back(chan);
          }
          ++fail;
        }
      }
    }
    return 1. * (ncx * ncy - fail) / (ncx * ncy);
  }  /// end of normal Test

  else  /// AS quality test !!!
  {
    if (me->kind() == MonitorElement::Kind::TH2F) {
      ncx = me->getTH2F()->GetXaxis()->GetNbins();
      ncy = me->getTH2F()->GetYaxis()->GetNbins();
      h = me->getTH2F();  // access Test histo
    } else if (me->kind() == MonitorElement::Kind::TH2S) {
      ncx = me->getTH2S()->GetXaxis()->GetNbins();
      ncy = me->getTH2S()->GetYaxis()->GetNbins();
      h = me->getTH2S();  // access Test histo
    } else if (me->kind() == MonitorElement::Kind::TH2D) {
      ncx = me->getTH2D()->GetXaxis()->GetNbins();
      ncy = me->getTH2D()->GetYaxis()->GetNbins();
      h = me->getTH2D();  // access Test histo
    } else {
      if (verbose_ > 0)
        std::cout << "QTest:ContentsWithinExpected AS"
                  << " ME does not contain TH2F/TH2S/TH2D, exiting\n";
      return -1;
    }

    // if (!rangeInitialized_) return 0; // all accepted if no initialization
    int fail = 0;
    for (int cx = 1; cx <= ncx; ++cx) {
      for (int cy = 1; cy <= ncy; ++cy) {
        bool failure = false;
        double Content = h->GetBinContent(h->GetBin(cx, cy));
        if (Content)
          failure = (Content < minMean_ || Content > maxMean_);
        if (failure)
          ++fail;
      }
    }
    return 1. * (ncx * ncy - fail) / (ncx * ncy);
  }  /// end of AS quality test
}
/// set expected value for mean
void ContentsWithinExpected::setMeanRange(double xmin, double xmax) {
  if (xmax < xmin)
    if (verbose_ > 0)
      std::cout << "QTest:ContentsWitinExpected"
                << " Illogical range: (" << xmin << ", " << xmax << "\n";
  minMean_ = xmin;
  maxMean_ = xmax;
  checkMean_ = true;
}

/// set expected value for mean
void ContentsWithinExpected::setRMSRange(double xmin, double xmax) {
  if (xmax < xmin)
    if (verbose_ > 0)
      std::cout << "QTest:ContentsWitinExpected"
                << " Illogical range: (" << xmin << ", " << xmax << "\n";
  minRMS_ = xmin;
  maxRMS_ = xmax;
  checkRMS_ = true;
}

//----------------------------------------------------------------//
//--------------------  MeanWithinExpected  ---------------------//
//---------------------------------------------------------------//
// run the test;
//   (a) if useRange is called: 1 if mean within allowed range, 0 otherwise
//   (b) is useRMS or useSigma is called: result is the probability
//   Prob(chi^2, ndof=1) that the mean of histogram will be deviated by more than
//   +/- delta from <expected_mean>, where delta = mean - <expected_mean>, and
//   chi^2 = (delta/sigma)^2. sigma is the RMS of the histogram ("useRMS") or
//   <expected_sigma> ("useSigma")
//   e.g. for delta = 1, Prob = 31.7%
//  for delta = 2, Prob = 4.55%
//   (returns result in [0, 1] or <0 for failure)
float MeanWithinExpected::runTest(const MonitorElement* me) {
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";

  if (minEntries_ != 0 && me->getEntries() < minEntries_)
    return -1;

  if (me->kind() == MonitorElement::Kind::TH1F) {
    h = me->getTH1F();  //access Test histo
  } else if (me->kind() == MonitorElement::Kind::TH1S) {
    h = me->getTH1S();  //access Test histo
  } else if (me->kind() == MonitorElement::Kind::TH1D) {
    h = me->getTH1D();  //access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:MeanWithinExpected"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1S/TH1D, exiting\n";
    return -1;
  }

  if (useRange_) {
    double mean = h->GetMean();
    if (mean <= xmax_ && mean >= xmin_)
      return 1;
    else
      return 0;
  } else if (useSigma_) {
    if (sigma_ != 0.) {
      double chi = (h->GetMean() - expMean_) / sigma_;
      return TMath::Prob(chi * chi, 1);
    } else {
      if (verbose_ > 0)
        std::cout << "QTest:MeanWithinExpected"
                  << " Error, sigma_ is zero, exiting\n";
      return 0;
    }
  } else if (useRMS_) {
    if (h->GetRMS() != 0.) {
      double chi = (h->GetMean() - expMean_) / h->GetRMS();
      return TMath::Prob(chi * chi, 1);
    } else {
      if (verbose_ > 0)
        std::cout << "QTest:MeanWithinExpected"
                  << " Error, RMS is zero, exiting\n";
      return 0;
    }
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:MeanWithinExpected"
                << " Error, neither Range, nor Sigma, nor RMS, exiting\n";
    return -1;
  }
}

void MeanWithinExpected::useRange(double xmin, double xmax) {
  useRange_ = true;
  useSigma_ = useRMS_ = false;
  xmin_ = xmin;
  xmax_ = xmax;
  if (xmin_ > xmax_)
    if (verbose_ > 0)
      std::cout << "QTest:MeanWithinExpected"
                << " Illogical range: (" << xmin_ << ", " << xmax_ << "\n";
}
void MeanWithinExpected::useSigma(double expectedSigma) {
  useSigma_ = true;
  useRMS_ = useRange_ = false;
  sigma_ = expectedSigma;
  if (sigma_ == 0)
    if (verbose_ > 0)
      std::cout << "QTest:MeanWithinExpected"
                << " Expected sigma = " << sigma_ << "\n";
}

void MeanWithinExpected::useRMS() {
  useRMS_ = true;
  useSigma_ = useRange_ = false;
}

//----------------------------------------------------------------//
//------------------------  CompareToMedian  ---------------------------//
//----------------------------------------------------------------//
/* 
Test for TProfile2D
For each x bin, the median value is calculated and then each value is compared with the median.
This procedure is repeated for each x-bin of the 2D profile
The parameters used for this comparison are:
MinRel and MaxRel to identify outliers wrt the median value
An absolute value (MinAbs, MaxAbs) on the median is used to identify a full region out of specification 
*/
float CompareToMedian::runTest(const MonitorElement* me) {
  int32_t nbins = 0, failed = 0;
  badChannels_.clear();

  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;

  if (verbose_ > 1) {
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";
    std::cout << "\tMin = " << _min << "; Max = " << _max << "\n";
    std::cout << "\tMinMedian = " << _minMed << "; MaxMedian = " << _maxMed << "\n";
    std::cout << "\tUseEmptyBins = " << (_emptyBins ? "Yes" : "No") << "\n";
  }

  if (me->kind() == MonitorElement::Kind::TPROFILE2D) {
    h = me->getTProfile2D();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:ContentsWithinExpected"
                << " ME does not contain TPROFILE2D, exiting\n";
    return -1;
  }

  nBinsX = h->GetNbinsX();
  nBinsY = h->GetNbinsY();
  int entries = 0;
  float median = 0.0;

  //Median calculated with partially sorted vector
  for (int binX = 1; binX <= nBinsX; binX++) {
    reset();
    // Fill vector
    for (int binY = 1; binY <= nBinsY; binY++) {
      int bin = h->GetBin(binX, binY);
      auto content = (double)h->GetBinContent(bin);
      if (content == 0 && !_emptyBins)
        continue;
      binValues.push_back(content);
      entries = me->getTProfile2D()->GetBinEntries(bin);
    }
    if (binValues.empty())
      continue;
    nbins += binValues.size();

    //calculate median
    if (!binValues.empty()) {
      int medPos = (int)binValues.size() / 2;
      nth_element(binValues.begin(), binValues.begin() + medPos, binValues.end());
      median = *(binValues.begin() + medPos);
    }

    // if median == 0, use the absolute cut
    if (median == 0) {
      if (verbose_ > 0) {
        std::cout << "QTest: Median is 0; the fixed cuts: [" << _minMed << "; " << _maxMed << "]  are used\n";
      }
      for (int binY = 1; binY <= nBinsY; binY++) {
        int bin = h->GetBin(binX, binY);
        auto content = (double)h->GetBinContent(bin);
        entries = me->getTProfile2D()->GetBinEntries(bin);
        if (entries == 0)
          continue;
        if (content > _maxMed || content < _minMed) {
          DQMChannel chan(binX, binY, 0, content, h->GetBinError(bin));
          badChannels_.push_back(chan);
          failed++;
        }
      }
      continue;
    }

    //Cut on stat: will mask rings with no enought of statistics
    if (median * entries < _statCut)
      continue;

    // If median is off the absolute cuts, declare everything bad (if bin has non zero entries)
    if (median > _maxMed || median < _minMed) {
      for (int binY = 1; binY <= nBinsY; binY++) {
        int bin = h->GetBin(binX, binY);
        auto content = (double)h->GetBinContent(bin);
        entries = me->getTProfile2D()->GetBinEntries(bin);
        if (entries == 0)
          continue;
        DQMChannel chan(binX, binY, 0, content / median, h->GetBinError(bin));
        badChannels_.push_back(chan);
        failed++;
      }
      continue;
    }

    // Test itself
    float minCut = median * _min;
    float maxCut = median * _max;
    for (int binY = 1; binY <= nBinsY; binY++) {
      int bin = h->GetBin(binX, binY);
      auto content = (double)h->GetBinContent(bin);
      entries = me->getTProfile2D()->GetBinEntries(bin);
      if (entries == 0)
        continue;
      if (content > maxCut || content < minCut) {
        DQMChannel chan(binX, binY, 0, content / median, h->GetBinError(bin));
        badChannels_.push_back(chan);
        failed++;
      }
    }
  }

  if (nbins == 0) {
    if (verbose_ > 0)
      std::cout << "QTest:CompareToMedian: Histogram is empty" << std::endl;
    return 1.;
  }
  return 1 - (float)failed / nbins;
}
//----------------------------------------------------------------//
//------------------------  CompareLastFilledBin -----------------//
//----------------------------------------------------------------//
/* 
Test for TH1F and TH2F
For the last filled bin the value is compared with specified upper and lower limits. If 
it is outside the limit the test failed test result is returned
The parameters used for this comparison are:
MinRel and MaxRel to check identify outliers wrt the median value
*/
float CompareLastFilledBin::runTest(const MonitorElement* me) {
  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h1 = nullptr;
  TH2* h2 = nullptr;
  if (verbose_ > 1) {
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";
    std::cout << "\tMin = " << _min << "; Max = " << _max << "\n";
  }
  if (me->kind() == MonitorElement::Kind::TH1F) {
    h1 = me->getTH1F();  // access Test histo
  } else if (me->kind() == MonitorElement::Kind::TH2F) {
    h2 = me->getTH2F();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:ContentsWithinExpected"
                << " ME does not contain TH1F or TH2F, exiting\n";
    return -1;
  }
  int lastBinX = 0;
  int lastBinY = 0;
  float lastBinVal;

  //--------- do the quality test for 1D histo ---------------//
  if (h1 != nullptr) {
    lastBinX = h1->FindLastBinAbove(_average, 1);
    lastBinVal = h1->GetBinContent(lastBinX);
    if (h1->GetEntries() == 0 || lastBinVal < 0)
      return 1;
  } else if (h2 != nullptr) {
    lastBinX = h2->FindLastBinAbove(_average, 1);
    lastBinY = h2->FindLastBinAbove(_average, 2);
    if (h2->GetEntries() == 0 || lastBinX < 0 || lastBinY < 0)
      return 1;
    lastBinVal = h2->GetBinContent(h2->GetBin(lastBinX, lastBinY));
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:" << getAlgoName() << " Histogram does not exist" << std::endl;
    return 1;
  }
  if (verbose_ > 0)
    std::cout << "Min and Max values " << _min << " " << _max << " Av value " << _average << " lastBinX " << lastBinX
              << " lastBinY " << lastBinY << " lastBinVal " << lastBinVal << std::endl;
  if (lastBinVal > _min && lastBinVal <= _max)
    return 1;
  else
    return 0;
}
//----------------------------------------------------//
//--------------- CheckVariance ---------------------//
//----------------------------------------------------//
float CheckVariance::runTest(const MonitorElement* me) {
  badChannels_.clear();

  if (!me)
    return -1;
  if (!me->getRootObject())
    return -1;
  TH1* h = nullptr;

  if (verbose_ > 1)
    std::cout << "QTest:" << getAlgoName() << "::runTest called on " << me->getFullname() << "\n";
  // -- TH1F
  if (me->kind() == MonitorElement::Kind::TH1F) {
    h = me->getTH1F();
  }
  // -- TH1D
  else if (me->kind() == MonitorElement::Kind::TH1D) {
    h = me->getTH1D();
  } else if (me->kind() == MonitorElement::Kind::TPROFILE) {
    h = me->getTProfile();  // access Test histo
  } else {
    if (verbose_ > 0)
      std::cout << "QTest:CheckVariance"
                << " ME " << me->getFullname() << " does not contain TH1F/TH1D/TPROFILE, exiting\n";
    return -1;
  }

  int ncx = h->GetXaxis()->GetNbins();

  double sum = 0;
  double sum2 = 0;
  for (int bin = 1; bin <= ncx; ++bin) {
    double contents = h->GetBinContent(bin);
    sum += contents;
  }
  if (sum == 0)
    return -1;
  double avg = sum / ncx;

  for (int bin = 1; bin <= ncx; ++bin) {
    double contents = h->GetBinContent(bin);
    sum2 += (contents - avg) * (contents - avg);
  }

  double Variance = TMath::Sqrt(sum2 / ncx);
  return Variance;
}
