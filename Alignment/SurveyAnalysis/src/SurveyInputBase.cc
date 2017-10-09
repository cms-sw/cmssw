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
  if (0 == theDetector)
    theDetector = comp;
  else
    theDetector->addComponent(comp);
}
