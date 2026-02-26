
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "QualityMetric.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#define GET_CLOCK_MONOTONIC(ts)                                      \
  {                                                                  \
    clock_serv_t cclock;                                             \
    mach_timespec_t mts;                                             \
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock); \
    clock_get_time(cclock, &mts);                                    \
    mach_port_deallocate(mach_task_self(), cclock);                  \
    ts.tv_sec = mts.tv_sec;                                          \
    ts.tv_nsec = mts.tv_nsec;                                        \
  }
#else
#define GET_CLOCK_MONOTONIC(ts) clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

using namespace XrdAdaptor;

QualityMetricWatch::QualityMetricWatch(QualityMetric *parent1, QualityMetric *parent2)
    : m_parent1(parent1), m_parent2(parent2) {
  // TODO: just assuming success.
  GET_CLOCK_MONOTONIC(m_start);
}

QualityMetricWatch::~QualityMetricWatch() {
  if (m_parent1 && m_parent2) {
    timespec stop;
    GET_CLOCK_MONOTONIC(stop);

    int ms = 1000 * (stop.tv_sec - m_start.tv_sec) + (stop.tv_nsec - m_start.tv_nsec) / 1e6;
    edm::LogVerbatim("XrdAdaptorInternal") << "Finished timer after " << ms << std::endl;
    m_parent1->finishWatch(stop, ms);
    m_parent2->finishWatch(stop, ms);
  }
}

QualityMetricWatch::QualityMetricWatch(QualityMetricWatch &&that) {
  m_parent1 = that.m_parent1;
  m_parent2 = that.m_parent2;
  m_start = that.m_start;
  that.m_parent1 = nullptr;
  that.m_parent2 = nullptr;
  that.m_start = {0, 0};
}

void QualityMetricWatch::swap(QualityMetricWatch &that) {
  QualityMetric *tmp;
  tmp = that.m_parent1;
  that.m_parent1 = m_parent1;
  m_parent1 = tmp;
  tmp = that.m_parent2;
  that.m_parent2 = m_parent2;
  m_parent2 = tmp;
  timespec tmp2;
  tmp2 = that.m_start;
  that.m_start = m_start;
  m_start = tmp2;
}

QualityMetric::QualityMetric(timespec now, int default_value)
    : m_value(default_value),
      m_interval0_n(0),
      m_interval0_val(-1),
      m_interval0_start(now.tv_sec),
      m_interval1_val(-1),
      m_interval2_val(-1),
      m_interval3_val(-1),
      m_interval4_val(-1) {}

void QualityMetric::finishWatch(timespec stop, int ms) {
  std::unique_lock<std::mutex> sentry(m_mutex);

  m_value = -1;
  if (stop.tv_sec > m_interval0_start + interval_length) {
    m_interval4_val = m_interval3_val;
    m_interval3_val = m_interval2_val;
    m_interval2_val = m_interval1_val;
    m_interval1_val = m_interval0_val;
    m_interval0_n = 1;
    m_interval0_val = ms;
    m_interval0_start = stop.tv_sec;
  } else {
    int num = m_interval0_val * m_interval0_n + ms;
    m_interval0_n++;
    m_interval0_val = num / m_interval0_n;
  }
}

unsigned QualityMetric::get() {
  std::unique_lock<std::mutex> sentry(m_mutex);

  if (m_value == -1) {
    unsigned den = 0;
    m_value = 0;
    if (m_interval0_val >= 0) {
      den += 16;
      m_value = 16 * m_interval0_val;
    }
    if (m_interval1_val >= 0) {
      den += 8;
      m_value += 8 * m_interval1_val;
    }
    if (m_interval2_val >= 0) {
      den += 4;
      m_value += 4 * m_interval2_val;
    }
    if (m_interval3_val >= 0) {
      den += 2;
      m_value += 2 * m_interval3_val;
    }
    if (m_interval4_val >= 0) {
      den += 1;
      m_value += m_interval4_val;
    }
    if (den)
      m_value /= den;
    else
      m_value = 260;
  }
  return m_value;
}

CMS_THREAD_SAFE QualityMetricFactory QualityMetricFactory::m_instance;

std::unique_ptr<QualityMetricSource> QualityMetricFactory::get(timespec now, const std::string &id) {
  auto itFound = m_instance.m_sources.find(id);
  if (itFound == m_instance.m_sources.end()) {
    // try to make a new one
    std::unique_ptr<QualityMetricUniqueSource> source(new QualityMetricUniqueSource(now));
    auto insertResult = m_instance.m_sources.insert(std::make_pair(id, source.get()));
    itFound = insertResult.first;
    if (insertResult.second) {  // Insert was successful; release our reference.
      source.release();
    }  // Otherwise, we raced with a different thread and they won; we will delete our new QM source.
  }
  return itFound->second->newSource(now);
}

QualityMetricSource::QualityMetricSource(QualityMetricUniqueSource &parent, timespec now, int default_value)
    : QualityMetric(now, default_value), m_parent(parent) {}

void QualityMetricSource::startWatch(QualityMetricWatch &watch) {
  QualityMetricWatch tmp(&m_parent, this);
  watch.swap(tmp);
}

QualityMetricUniqueSource::QualityMetricUniqueSource(timespec now) : QualityMetric(now) {}

std::unique_ptr<QualityMetricSource> QualityMetricUniqueSource::newSource(timespec now) {
  std::unique_ptr<QualityMetricSource> child(new QualityMetricSource(*this, now, get()));
  return child;
}
