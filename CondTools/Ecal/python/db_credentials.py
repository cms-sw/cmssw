#import FWCore.ParameterSet.Config as cms
import CondCore.Utilities.credentials as credentials
import socket

db_service = 'cms_omds_adg'
if socket.getfqdn().strip().endswith('.cms'):
    db_service = 'cms_omds_lb'

def get_db_credentials( db_account ):
    machine = '%s/%s' %(db_service, db_account)
    creds = credentials.get_credentials( machine )
    if not creds is None:
        (username, account, pwd) = creds
        return db_service, username, pwd
    else:
        raise Exception('Entry for service %s not found in .netrc' %machine )
