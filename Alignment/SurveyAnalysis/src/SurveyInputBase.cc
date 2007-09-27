#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

Alignable* SurveyInputBase::theDetector(0);

SurveyInputBase::~SurveyInputBase()
{
  delete theDetector;

  theDetector = 0;
}

void SurveyInputBase::addComponent(Alignable* comp)
{
  delete theDetector;

  theDetector = comp;
}
