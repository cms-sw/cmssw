#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <math.h>
#include <stdlib.h>
#include <ctime>

#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/Utilities/interface/isFinite.h"

/////////////////////////////////////////////////////////////
//
// This test exercises the MonitoredQuantity class
//
/////////////////////////////////////////////////////////////

using namespace stor;


class testMonitoredQuantity : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testMonitoredQuantity);
  CPPUNIT_TEST(testEmpty);
  CPPUNIT_TEST(testFull);
  CPPUNIT_TEST(testRecent);
  CPPUNIT_TEST(testDisable);

  CPPUNIT_TEST_SUITE_END();

public:
  testMonitoredQuantity();
  void testEmpty();
  void testFull();
  void testRecent();
  void testDisable();

private:

  void accumulateSamples
  (
    unsigned int sampleCount,
    double &squareSum
  );

  void testResults
  (
    MonitoredQuantity::DataSetType type,
    unsigned int cycleCount,
    unsigned int sampleCount,
    double squareSum
  );


  MonitoredQuantity quantity_;

  const double multiplier_;
};

testMonitoredQuantity::testMonitoredQuantity() :
quantity_(boost::posix_time::milliseconds(1),boost::posix_time::milliseconds(2)),
//Only 2 bins deep history for testing, allow fast updates
multiplier_(drand48()*100)
{
  srand48(time(0));
}

void testMonitoredQuantity::accumulateSamples
(
  unsigned int sampleCount,
  double &squareSum
)
{
  assert(edm::isFinite(squareSum));
  for (
    unsigned int i = 1;
    i <= sampleCount;
    ++i
  )
  {
    quantity_.addSample(i*multiplier_);
    squareSum += pow(i*multiplier_,2);
    ::usleep(1000);
  }
  quantity_.calculateStatistics();
}


void testMonitoredQuantity::testResults
(
  MonitoredQuantity::DataSetType type,
  unsigned int cycleCount,
  unsigned int sampleCount,
  double squareSum
)
{
  // we don't expect an exact agreement due to rounding precision
  const double smallValue = 1e-05;;
  MonitoredQuantity::Stats stats;
  quantity_.getStats(stats);

  CPPUNIT_ASSERT(stats.getSampleCount(type) == cycleCount * sampleCount);
  
  CPPUNIT_ASSERT(
    fabs(
      stats.getValueSum(type) -
      cycleCount * static_cast<double>(sampleCount)*(sampleCount+1)/2 * multiplier_
    ) < smallValue);

  CPPUNIT_ASSERT(
    fabs(
      stats.getValueAverage(type) -
      ((cycleCount) ? static_cast<double>(sampleCount+1)/2 * multiplier_ : 0)
    ) < smallValue);

  CPPUNIT_ASSERT(stats.getValueMin(type) == 
    (cycleCount) ? multiplier_ : 1e+9);

  CPPUNIT_ASSERT(stats.getValueMax(type) == 
    (cycleCount) ? static_cast<double>(sampleCount)*multiplier_ : 1e-9);

  const double duration = utils::durationToSeconds(stats.getDuration(type));
  if (duration > 0)
  {
    CPPUNIT_ASSERT(
      fabs(
        stats.getSampleRate(type) -
        cycleCount*sampleCount/duration
      ) < smallValue);    

    CPPUNIT_ASSERT(
      fabs(
        stats.getSampleLatency(type) -
        1e6*duration/(cycleCount*sampleCount)
      ) < smallValue);    

    CPPUNIT_ASSERT(
      fabs(
        stats.getValueRate(type) -
        stats.getValueSum(type)/duration
      ) < smallValue);
  }

  if (sampleCount > 0)
  {
    unsigned long numEntries = cycleCount * sampleCount;
    double rmsSquared = pow(stats.getValueRMS(type), 2);
    double expectedSquare = squareSum/numEntries 
      - pow(stats.getValueAverage(type),2);

    double difference = rmsSquared - expectedSquare;
    std::cerr << "\n type " << type;
    std::cerr << "\n square sum " << squareSum;
    std::cerr << "\n cycle count " << cycleCount;
    std::cerr << "\n sample count " << sampleCount;
    std::cerr << "\n RMS:        " << stats.getValueRMS(type);
    std::cerr << "\n expected square: " << expectedSquare;
    std::cerr << "\n difference: " << difference;

    std::cerr << '\n';

    CPPUNIT_ASSERT(fabs(difference) < smallValue);
//     CPPUNIT_ASSERT(
//       fabs(
//            stats.getValueRMS(type) -
//            sqrt((squareSum/(cycleCount*sampleCount))-pow(stats.getValueAverage(type),2))
//            ) < smallValue);
  }
}


void testMonitoredQuantity::testEmpty()
{
  quantity_.reset();

  quantity_.calculateStatistics();

  testResults(MonitoredQuantity::FULL, 0, 0, 0);
  testResults(MonitoredQuantity::RECENT, 0, 0, 0);
}


void testMonitoredQuantity::testFull()
{
  int sampleCount = 100;
  double squareSum = 0.0;

  quantity_.reset();

  accumulateSamples(sampleCount, squareSum);

  testResults(MonitoredQuantity::FULL, 1, sampleCount, squareSum);
}


void testMonitoredQuantity::testRecent()
{
  int sampleCount = 50;
  double squareSum=0.0, totalSquareSum=0.0;

  quantity_.reset();

  accumulateSamples(sampleCount, squareSum);
  // reset square sum as buffer is only 2 deep
  totalSquareSum = squareSum;
  squareSum = 0;
  accumulateSamples(sampleCount, squareSum);
  accumulateSamples(sampleCount, squareSum);
  totalSquareSum += squareSum;

  testResults(MonitoredQuantity::FULL, 3, sampleCount, totalSquareSum);
  testResults(MonitoredQuantity::RECENT, 2, sampleCount, squareSum);
}


void testMonitoredQuantity::testDisable()
{
  int sampleCount = 50;
  double squareSum(0.0), dummySquareSum(0.0);

  quantity_.reset();

  accumulateSamples(sampleCount, squareSum);
  // disable the quantity, no changes expected
  quantity_.disable();
  accumulateSamples(sampleCount, dummySquareSum);

  testResults(MonitoredQuantity::FULL, 1, sampleCount, squareSum);
  testResults(MonitoredQuantity::RECENT, 1, sampleCount, squareSum);

  // Reenable quantity. This resets everything.
  quantity_.enable();
  squareSum = 0;
  accumulateSamples(sampleCount, squareSum);

  testResults(MonitoredQuantity::FULL, 1, sampleCount, squareSum);
  testResults(MonitoredQuantity::RECENT, 1, sampleCount, squareSum);
}




// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testMonitoredQuantity);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
