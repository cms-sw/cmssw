import unittest
import sys
import os

if __name__ == '__main__':    
    os.chdir(os.environ['HEPPY'])
    suites = []
    
    pcks = [
        'analyzers',
        'display', 
        'framework',
        'test',  # if particles is before test, test fails! 
        'papas', 
        'particles',
        'scripts', 
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

 

