import netrc
import os

netrcFileName = '.netrc'

def get_credentials_from_file( service, authFile=None ):
    if not authFile is None:
        if os.path.isdir(authFile):
            authFile = os.path.join( authFile, netrcFileName )
    creds = netrc.netrc( authFile ).authenticators(service)
    return creds

def get_credentials( authPathEnvVar, service, authFile=None ):
    if authFile is None:
        if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
            authFile = os.path.join(authPath, netrcFileName)
    return get_credentials_from_file( service, authFile )

