import CondCore.Utilities.credentials as credentials
import socket

authPathEnvVar = 'COND_AUTH_PATH'
db_service = 'cms_omds_adg'
if socket.getfqdn().strip().endswith('.cms'):
    db_service = 'cms_omds_lb'
db_machine = db_service + '/CMS_ECAL_R' 


def get_readOnly_db_credentials():
    creds = credentials.get_credentials( authPathEnvVar, db_machine )
    if not creds is None:
        (username, account, pwd) = creds
        return db_service, username, pwd
    else:
        raise Exception('Entry for service %s not found in .netrc' %db_machine )
