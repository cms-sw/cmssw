from Configuration.PyReleaseValidation.upgradeWorkflowComponents import upgradeProperties as properties
from Configuration.AlCa.autoCond import autoCond
from Configuration.StandardSequences.Eras import eras

DEFAULT_VERSION = "Run4D110"

def get_era_and_conditions(version_key):
    """Retrieve the era and global tag for a given version key.

    Args:
        version_key (str): The version key to look up.

    Returns:
        tuple: A tuple containing the global tag and era object.

    Raises:
        KeyError: If the version key or global tag is not found.
    """
    # Ensure the version key exists in the properties for Run4
    if version_key not in properties['Run4']:
        raise KeyError(f"Version key '{version_key}' not found in properties['Run4'].")

    # Retrieve the global tag key
    global_tag_key = properties['Run4'][version_key]['GT']
    print(f"Global tag key from properties: {global_tag_key}")

    # Validate the existence of the global tag in autoCond
    global_tag_name = global_tag_key.replace("auto:", "")
    if global_tag_name not in autoCond:
        raise KeyError(f"Global tag key '{global_tag_key}' not found in autoCond.")

    # Retrieve the era key and get the corresponding era object
    era_key = properties['Run4'][version_key]['Era']
    print(f"Constructed era key from properties: {era_key}")
    era = getattr(eras, era_key)

    return global_tag_key, era
