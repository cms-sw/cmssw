#!/bin/bash
grep -B 1 -A 15 'IETA" type="int">18' CRUZETPhysicsV4T*xml | grep -B 1 -A 14 'IPHI" type="int">5<' | grep -B 5 -A 10 'LUT_TYPE" type="int">1' | grep -B 14 -A 1 'DEPTH" type="int">1<'
