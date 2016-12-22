import copy
from links import Links
from distance import distance
from ROOT import TVector3


def merge_clusters(elements, layer):
    merged = []
    elem_in_layer = []
    elem_other = []
    for elem in elements:
        if elem.layer == layer:
            elem_in_layer.append(elem)
        else:
            elem_other.append(elem)
    links = Links(elem_in_layer, distance)
    for group in links.groups.values():
        if len(group) == 1:
            merged.append(group[0]) 
            continue
        supercluster = None
        for cluster in group: 
            if supercluster is None:
                supercluster = copy.copy(cluster)
                merged.append(supercluster)
                continue
            else: 
                supercluster += cluster
    merged.extend(elem_other)
    return merged
