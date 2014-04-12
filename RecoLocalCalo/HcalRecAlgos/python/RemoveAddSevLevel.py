import FWCore.ParameterSet.Config as cms


def RemoveFlag(sevLevelComputer,flag="HFLongShort"):
    ''' Removes the specified flag from the Severity Level Computer,
    and returns the revised Computer.'''
    
    removeSeverity=-1 # Track which Severity Level has been modified



    # Loop over all levels
    for i in range(len(sevLevelComputer.SeverityLevels)):
        Flags=sevLevelComputer.SeverityLevels[i].RecHitFlags.value()
        if flag not in Flags:  # Flag not present for this level
            continue
        #Remove flag
        Flags.remove(flag)
        ChanStat=sevLevelComputer.SeverityLevels[i].ChannelStatus.value()
        # Check to see if Severity Level no longer contains any useful information
        if len(Flags)==0 and ChanStat==['']:
            removeSeverity=i
        else:
            # Set revised list of flags for this severity level
            sevLevelComputer.SeverityLevels[i].RecHitFlags=Flags
        break

    # Removing flag results in empty severity level; remove it
    if (removeSeverity>-1):
        sevLevelComputer.SeverityLevels.remove(sevLevelComputer.SeverityLevels[removeSeverity])

    return sevLevelComputer


def PrintLevels(SLComp):
    print "Severity Level Computer Levels and associated flags/Channel Status values:"
    for i in SLComp.SeverityLevels:
        print "\t Level = %i"%i.Level.value()
        print "\t\t RecHit Flags = %s"%i.RecHitFlags.value()
        print "\t\t Channel Status = %s"%i.ChannelStatus.value()
        print
    return


def AddFlag(sevLevelComputer,flag="UserDefinedBit0",severity=10):
    ''' Adds specified flag to severity level computer using specified severity level.
        If flag already exists at another severity level, it is removed from that level.
        '''
    
    AddedSeverity=False
    removeSeverity=-1

    allowedflags=[]
    for i in sevLevelComputer.SeverityLevels:
        for j in i.RecHitFlags.value():
            if j=="":
                continue
            allowedflags.append(j)
            
    #print "Allowed flags = ",allowedflags
    if flag not in allowedflags:
        print "\n\n"
        for j in range(0,3):
            print "###################################################"
        print "\nWARNING!!!!!! You are adding a flag \n\t'%s' \nthat is not defined in the Severity Level Computer!"%flag
        print "This can be EXCEPTIONALLY dangerous if you do not \nknow what you are doing!\n"
        print "Proceed with EXTREME caution!\n"
        for j in range(0,3):
            print "###################################################"
        print "\n\n"

    #Loop over severity Levels
    for i in range(len(sevLevelComputer.SeverityLevels)):
        Level=sevLevelComputer.SeverityLevels[i].Level.value()
        Flags=sevLevelComputer.SeverityLevels[i].RecHitFlags.value()
        if Level==severity:  # Found the specified level
            if (Flags==['']):
                Flags=[flag] # Create new vector for this flag
            else:
                if flag not in Flags:  # don't need to add flag if it's already there
                    Flags.append(flag) # append flag to existing vector
            sevLevelComputer.SeverityLevels[i].RecHitFlags=Flags  # Set new RecHitFlags vector
            AddedSeverity=True
        else:  # Found some other level; be sure to remove flag from it
            if flag not in Flags:
                continue
            else:
                Flags.remove(flag)
                # Removing flag leaves nothing else:  need to remove this level completely
                if len(Flags)==0 and ChanStat==['']:
                    removeSeverity=i
                else:
                    sevLevelComputer.SeverityLevels[i].RecHitFlags=Flags

    # Remove any newly-empty levels
    if (removeSeverity>-1):
        sevLevelComputer.SeverityLevels.remove(sevLevelComputer.SeverityLevels[removeSeverity])

    # No existing severity level for specified severity was found;
    # add a new one
    if (AddedSeverity==False):
        sevLevelComputer.SeverityLevels.append(cms.PSet(Level=cms.int32(severity),
                                                        RecHitFlags=cms.vstring(flag),
                                                        ChannelStatus=cms.vstring("")))
    return sevLevelComputer


    
##########################

if __name__=="__main__":
    import hcalRecAlgoESProd_cfi as ES
    ES.hcalRecAlgos=RemoveFlag(ES.hcalRecAlgos)
    ES.hcalRecAlgos=AddFlag(ES.hcalRecAlgos,flag="HOBit",severity=5)
    PrintLevels(ES.hcalRecAlgos)

