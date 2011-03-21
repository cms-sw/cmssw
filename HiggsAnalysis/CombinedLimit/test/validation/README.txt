Instructions to run the test suite:

0) ./testSuite list will list possible validation jobs

1) ./testSuite create will create the jobs
  you can re-create part of the jobs without affecting the others
  note that re-creation does not delete old outputs

2a) ./testSuite run 
   will run jobs locally.
   use option -j N to run on N processors

2b) ./testSuite submit 
   will submit jobs to lxbatch
   (must be running from AFS)

3) ./testSuite report [ -r reference.json ] [ -f text|twiki ]
   will read results and produce json file in directory,
   and will print out results

4) ./restSuite print [-r reference.json ]  -f text|twiki ]
   will just print out results, without reading output rootfiles again

5) to update some reference numbers or add new ones, use whatever diff editor 
   to compare reference.json with <dir>/report.json and propagate changes

6) to add new tests, edit test_XX.py files

Note: you can combine commands in a single line, e.g.
 ./testSuite create run -j 4 report
