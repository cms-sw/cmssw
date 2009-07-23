#!/bin/bash

for i in `seq 1 100`
do
    fn="out_files_$i.tgz"
    tar -xzf $fn
done

