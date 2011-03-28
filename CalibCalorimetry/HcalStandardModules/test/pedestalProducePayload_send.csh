#!/bin/csh

set currentDir = `pwd`

bsub -q cmscaf1nd pedestalProducePayload_batch.csh $currentDir
