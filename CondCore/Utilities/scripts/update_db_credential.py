#!/usr/bin/env python
import sys
import getpass
import subprocess
import optparse

def update_credential( serviceName, accountName, newPassword ):
    command = 'cmscond_authentication_manager --update_conn -s %s -u %s -p %s' %(serviceName, accountName, newPassword )
    pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    stdout_val = pipe.communicate()[0]
    return stdout_val
 
def main():

    parser = optparse.OptionParser(usage =
        'Usage: %prog <file> [<file> ...]\n'
    )

    parser.add_option('-s', '--service',
        dest = 'service',
        help = 'the service hosting the account'
    )

    parser.add_option('-a','--accountName',
        dest = 'accountName',
        help = 'the account name to change'
    )

    (options, arguments) = parser.parse_args()
  
    if options.service == None or options.accountName == None:
        parser.print_help()
        return -2
    
    password = getpass.getpass( prompt= 'Enter the new password:')

    try:
        print update_credential( options.service, options.accountName, password )
        print 'Password changed.' 
    except e:
        print 'Update credential failed.' 
if __name__ == '__main__':
    sys.exit(main())
