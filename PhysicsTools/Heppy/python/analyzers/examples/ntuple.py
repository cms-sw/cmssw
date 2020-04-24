#!/bin/env python

def var( tree, varName, type=float ):
    tree.var(varName, type)

def fill( tree, varName, value ):
    tree.fill( varName, value )

# event information

def bookEvent(tree): 
    var(tree, 'run')
    var(tree, 'lumi')
    var(tree, 'event')
 
def fillEvent(tree, event):
    fill(tree, 'run', event.run)
    fill(tree, 'lumi', event.lumi)
    fill(tree, 'event', event.eventId)


# simple particle

def bookParticle( tree, pName ):
    var(tree, '{pName}_pdgid'.format(pName=pName))
    var(tree, '{pName}_e'.format(pName=pName))
    var(tree, '{pName}_pt'.format(pName=pName))
    var(tree, '{pName}_eta'.format(pName=pName))
    var(tree, '{pName}_phi'.format(pName=pName))
    var(tree, '{pName}_m'.format(pName=pName))  
    var(tree, '{pName}_q'.format(pName=pName))

def fillParticle( tree, pName, particle ):
    fill(tree, '{pName}_pdgid'.format(pName=pName), particle.pdgId() )
    fill(tree, '{pName}_e'.format(pName=pName), particle.energy() )
    fill(tree, '{pName}_pt'.format(pName=pName), particle.pt() )
    fill(tree, '{pName}_eta'.format(pName=pName), particle.eta() )
    fill(tree, '{pName}_phi'.format(pName=pName), particle.phi() )
    fill(tree, '{pName}_m'.format(pName=pName), particle.mass() )
    fill(tree, '{pName}_q'.format(pName=pName), particle.charge() )

def bookMet(tree, pName):
    var(tree, '{pName}_pt'.format(pName=pName))
    var(tree, '{pName}_phi'.format(pName=pName))
    var(tree, '{pName}_sumet'.format(pName=pName))

def fillMet(tree, pName, met):
    fill(tree, '{pName}_pt'.format(pName=pName), met.pt())
    fill(tree, '{pName}_phi'.format(pName=pName), met.phi())
    fill(tree, '{pName}_sumet'.format(pName=pName), met.sumEt())

def bookGenTau(tree, pName, pfdiscs, calodiscs):
    bookJet(tree, pName)   
    bookTau(tree, '{pName}_calo'.format(pName=pName), calodiscs)
    bookTau(tree, '{pName}_pf'.format(pName=pName), pfdiscs)
    bookJet(tree, '{pName}_pfjet'.format(pName=pName))

def fillGenTau(tree, pName, tau):
    fillJet(tree, pName, tau)   
    fillTau(tree, '{pName}_calo'.format(pName=pName), tau.match_calo)
    fillTau(tree, '{pName}_pf'.format(pName=pName), tau.match_pf)
    fillJet(tree, '{pName}_pfjet'.format(pName=pName), tau.match_pfjet)


def bookTau(tree, pName, discNames):
    bookParticle(tree, pName)   
    var(tree, '{pName}_nsigcharged'.format(pName=pName))
    var(tree, '{pName}_isolation'.format(pName=pName))
    for discName in discNames:
        var(tree, '{pName}_{disc}'.format(pName=pName,
                                          disc=discName))
        
def fillTau(tree, pName, tau):
    if not tau: return 
    fillParticle(tree, pName, tau)
    fill(tree, '{pName}_nsigcharged'.format(pName=pName), len(tau.signalCharged()))
    fill(tree, '{pName}_isolation'.format(pName=pName), tau.isolation())
    for discName, value in tau.discs.iteritems():
        fill(tree, '{pName}_{disc}'.format(pName=pName,
                                           disc=discName), value)


# jet

def bookComponent( tree, pName ):
    var(tree, '{pName}_e'.format(pName=pName))
    var(tree, '{pName}_pt'.format(pName=pName))
    var(tree, '{pName}_num'.format(pName=pName))

def fillComponent(tree, pName, component):
    fill(tree, '{pName}_e'.format(pName=pName), component.e() )
    fill(tree, '{pName}_pt'.format(pName=pName), component.pt() )
    fill(tree, '{pName}_num'.format(pName=pName), component.num() )
    
    
pdgids = [211, 22, 130, 11, 13]
    
def bookJet( tree, pName ):
    bookParticle(tree, pName )
    for pdgid in pdgids:
        bookComponent(tree, '{pName}_{pdgid:d}'.format(pName=pName, pdgid=pdgid))
    # var(tree, '{pName}_npart'.format(pName=pName))

def fillJet( tree, pName, jet ):
    if not jet: return
    fillParticle(tree, pName, jet )
    for pdgid in pdgids:
        component = jet.constituents.get(pdgid, None)
        if component is not None:
            fillComponent(tree,
                          '{pName}_{pdgid:d}'.format(pName=pName, pdgid=pdgid),
                          component )
        else:
            import pdb; pdb.set_trace()
            print jet

