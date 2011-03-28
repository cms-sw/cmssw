#!/bin/csh

set currentDir = `pwd`

bsub -q cmscaf1nd -J job1 < pedestalProducePayload_batch.csh $currentDir
