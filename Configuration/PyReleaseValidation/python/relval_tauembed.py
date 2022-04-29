from  Configuration.PyReleaseValidation.relval_steps import *

workflows = Matrix()

workflows[100.0] = ['', ['RunDoubleMuonTE2016C', 'RAWRECOTE16', 'RAWRECOLHECLEANTE16', 'EMBEDHADTE16']]
