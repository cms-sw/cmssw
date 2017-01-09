import netrc
import os

def get_credentials_from_file( service, authFile=None ):
    creds = netrc.netrc( authFile ).authenticators(service)
    return creds

def get_credentials( authPathEnvVar, service, authFile=None ):
    if authFile is None:
        if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
            authFile = os.path.join(authPath,'.netrc')
    return get_credentials_from_file( service, authFile )

