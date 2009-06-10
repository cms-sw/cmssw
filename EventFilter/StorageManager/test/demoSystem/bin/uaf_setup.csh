if (`hostname` =~ *srv-C*) then
    source /nfshome0/cmssw2/scripts/setup.csh
else
    if (`hostname` =~ *fnal.gov*) then
        source /uscmst1/prod/sw/cms/cshrc uaf
    endif
endif
