#!/bin/bash


DIRBASE="$CMSSW_BASE/src/Alignment/APEEstimation"

mv $DIRBASE/hists/plots/ideal/ $DIRBASE/hists/plots/ideal_old/
mv $DIRBASE/hists/plots/data/ $DIRBASE/hists/plots/data_old/

mkdir $DIRBASE/hists/plots/
mkdir $DIRBASE/hists/plots/ideal/
mkdir $DIRBASE/hists/plots/data/


root -l -b $DIRBASE/macros/commandsApeOverview.C

ps2pdf $DIRBASE/hists/plots/test1.ps $DIRBASE/hists/plots/test1.pdf
ps2pdf $DIRBASE/hists/plots/test2.ps $DIRBASE/hists/plots/test2.pdf
ps2pdf $DIRBASE/hists/plots/test3.ps $DIRBASE/hists/plots/test3.pdf
ps2pdf $DIRBASE/hists/plots/testSummary.ps $DIRBASE/hists/plots/testSummary.pdf

rm $DIRBASE/hists/plots/*.ps

mv $DIRBASE/hists/plots/test*.pdf $DIRBASE/hists/plots/ideal/.



root -l -b $DIRBASE/macros/commandsApeOverviewData.C

ps2pdf $DIRBASE/hists/plots/test1.ps $DIRBASE/hists/plots/test1.pdf
ps2pdf $DIRBASE/hists/plots/test2.ps $DIRBASE/hists/plots/test2.pdf
ps2pdf $DIRBASE/hists/plots/test3.ps $DIRBASE/hists/plots/test3.pdf
ps2pdf $DIRBASE/hists/plots/testSummary.ps $DIRBASE/hists/plots/testSummary.pdf

rm $DIRBASE/hists/plots/*.ps

mv $DIRBASE/hists/plots/test*.pdf $DIRBASE/hists/plots/data/.



