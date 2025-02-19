
import re

validNuisancePatterns = [
    {'prefix':'lumi', 'remainder':''},
    {'prefix':'pdf', 'remainder':'_(qqbar|gg|qg)'},
    {'prefix':'QCDscale', 'remainder':'_\w+'},
    {'prefix':'UEPS', 'remainder':''},
    {'prefix':'FakeRate', 'remainder':''},
    {'prefix':'CMS', 'remainder':'_(eff|fake|trigger|scale|res)_([gemtjb]|met)'},
    ]


def isValidNuisanceName(name):
    for pattern in validNuisancePatterns:
        prefixMatch = re.match(pattern['prefix']+'(.*)', name)
        if prefixMatch:
            remainder = prefixMatch.groups()[0]
            if not re.match(pattern['remainder'], remainder):
                return False
    else:
        return True

if __name__ == "__main__":
    from pprint import pprint
    names = ['CMS_eff_j', 'CMS_eff_k', 'CMS_p_scale_e', 'lumi', 'QCDscale']
    results = map(validateNuisance, names)
    pprint(zip(names, results))
