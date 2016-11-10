import unittest
import sys
import os

if __name__ == '__main__':   

    heppy_path = '/'.join([os.environ['CMSSW_BASE'], 
                           'src/PhysicsTools/HeppyCore/python']) 
    os.chdir(heppy_path)

    suites = []
    
    pcks = [
        'analyzers',
        'display', 
        'framework',
        'test',  # if particles is before test, test fails! 
        'papas', 
        'particles',
        'statistics',
        'utils'
        ]

    for pck in pcks:
        suites.append(unittest.TestLoader().discover(pck))

    suite = unittest.TestSuite(suites)
    # result = unittest.TextTestResult(sys.stdout, True, 1)
    # suite.run(result)
    runner = unittest.TextTestRunner()
    runner.run(suite)

 

